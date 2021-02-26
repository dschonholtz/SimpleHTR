import argparse
import json
import glob

import cv2
import numpy as np
import editdistance
from path import Path
from fuzzywuzzy import process

from DataLoaderIAM import DataLoaderIAM, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnSummary = '../model/summary.json'
    fnInfer = '../data/*.png'
    fnCorpus = '../data/corpus.txt'


def write_summary(charErrorRates, wordAccuracies):
    with open(FilePaths.fnSummary, 'w') as f:
        json.dump({'charErrorRates': charErrorRates, 'wordAccuracies': wordAccuracies}, f)


def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    summaryCharErrorRates = []
    summaryWordAccuracies = []
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 25  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print(f'Epoch: {epoch} Batch: {iterInfo[0]}/{iterInfo[1]} Loss: {loss}')

        # validate
        charErrorRate, wordAccuracy = validate(model, loader)

        # write summary
        summaryCharErrorRates.append(charErrorRate)
        summaryWordAccuracies.append(wordAccuracy)
        write_summary(summaryCharErrorRates, summaryWordAccuracies)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {charErrorRate * 100.0}%')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print(f'No more improvement since {earlyStopping} epochs. Training stopped.')
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print(f'Batch: {iterInfo[0]} / {iterInfo[1]}')
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print(f'Character error rate: {charErrorRate * 100.0}%. Word accuracy: {wordAccuracy * 100.0}%.')
    return charErrorRate, wordAccuracy

def remove_horiz_lines(image):
    """ Clean up the image by removing horizontal lines """
    orig_image = np.array(image)
    rtn, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    detected_h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel,
                                      iterations=1)
    cnts = cv2.findContours(detected_h_lines, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.drawContours(image, cnts, -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel,
                                    iterations=1)
    show_image = False
    if show_image:
        for wnd in ['image', 'orig_image', 'thresh', 'detected_h_lines', 'result']:
            cv2.namedWindow(wnd, cv2.WINDOW_NORMAL)
            w, h = image.shape
            cv2.resizeWindow(wnd, h * 3, w * 3)

        cv2.imshow('orig_image', orig_image)
        cv2.imshow('thresh', thresh)
        cv2.imshow('detected_h_lines', detected_h_lines)
        cv2.imshow('image', image)
        cv2.imshow('result', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return result

def crop(img, name):
    """ Find the potential text area an split into works. This returns
    and array of images where each image should be a single word
    """
    show_image = False
    img = remove_horiz_lines(img)

    # find the outlines for the 'text'
    ret, threshold = cv2.threshold(img, 176, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # for testing draw the contours on the debug image
    c_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(c_img, contours, -1, (0, 255, 0))

    # for each contour check if it could be text. If not white out the contour
    # in the image.
    v_center = img.shape[0] / 2
    rects = []
    for idx, cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        # a contour is text if it crosses the vertical center of the image
        # and is not within the left 'oval' edge (Note: this only works for
        # left ovals
        if y < v_center < y + h - 5 and x + w > 50:
            # found potential text so add to rectangles to consider
            rects.append([x, y, x+w, y+h])
            cv2.rectangle(c_img, (x,y), (x+w,y+h), (255,0,0))
        else:
            # else white out the contour area
            cv2.drawContours(c_img, contours, idx, (255, 255, 255), -1)
            cv2.drawContours(img, contours, idx, (255, 255, 255), -1)

    # soft the rectangles left to right and then group into works, i.e. move
    # to next 'group' (i.e. word) if the gap between the rectangles is large
    # enough to be a word break.
    rects.sort(key=lambda x: x[0])
    grouped_rects = [rects.pop(0)]
    for t_rect in rects:
        cur_rect = grouped_rects[-1]
        if t_rect[0] > cur_rect[2] + 5:
            grouped_rects.append(t_rect)
        else:
            cur_rect[1] = min(cur_rect[1], t_rect[1])
            cur_rect[2] = max(cur_rect[2], t_rect[2])
            cur_rect[3] = max(cur_rect[3], t_rect[3])

    # clip the workds into their own images
    images = []
    for g_rect in grouped_rects:
        cv2.rectangle(c_img, (g_rect[0],g_rect[1]), (g_rect[2], g_rect[3]), (0,0,255))
        images.append(img[g_rect[1]:g_rect[3], g_rect[0]:g_rect[2]])

    if show_image:
        for wnd in ['orig_image', 'threshold', 'image']:
            cv2.namedWindow(wnd, cv2.WINDOW_NORMAL)
            w, h = img.shape
            cv2.resizeWindow(wnd, h * 3, w * 3)
        cv2.imshow('orig_image', img)
        cv2.imshow('threshold', threshold)

        cv2.imshow('image', c_img)
        cv2.waitKey(0)
    cv2.imwrite(name, c_img)

    return images


def infer_set(model, filepath, names=()):
    "recognize text in image provided by file path"
    out_names = []
    out_probs = []
    for fname in glob.glob(filepath):
        if '-out.jpg' not in fname:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            print(img.shape)
            out_name = fname.replace('.jpg', '-out.jpg')
            imgs = crop(img, out_name)
            name = ''
            prop = 1
            for img in imgs:
                (recognized, probability) = infer(model, img)
                name += recognized[0] + ' '
                prop *= probability[0]
            if names:
                (match, score) = process.extractOne(name, names)
            else:
                match = ''
                score = 0
            out_names.append(name)
            out_probs.append(probability)
            print(fname, round(prop * 100, 2), name, score, match)
    return out_names, out_probs


def infer(model, img):
    img = preprocess(img, Model.imgSize)
    show_window = False
    if show_window:
        timg = (img + 0.5) * 255
        timg = timg.astype(np.uint8)
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        w, h = timg.shape
        cv2.resizeWindow('test', h * 4, w * 4)
        cv2.imshow('test', timg)
        cv2.waitKey(0)
    batch = Batch(None, [img])
    return model.inferBatch(batch, True)


def main():
    "main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath',
                        help='CTC decoder')
    parser.add_argument('--batch_size', help='batch size', type=int, default=100)
    parser.add_argument('--data_dir', help='directory containing IAM dataset', type=Path, required=False)
    parser.add_argument('--fast', help='use lmdb to load images', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    args = parser.parse_args()

    # set chosen CTC decoder
    if args.decoder == 'bestpath':
        decoderType = DecoderType.BestPath
    elif args.decoder == 'beamsearch':
        decoderType = DecoderType.BeamSearch
    elif args.decoder == 'wordbeamsearch':
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, Model.imgSize, Model.maxTextLen, args.fast)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # infer text on test image
    else:
        names = [
            'Bobby Bluestocking',
            'Candice Competent',
            'Honest Jim',
            'Candidate 3(Contest 1)',
            'Indie Pendant',
            'Jim Bell',
            'Polyanne Precious'
        ]
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        data = infer_set(model, FilePaths.fnInfer, names)
        print(data)

if __name__ == '__main__':
    main()
