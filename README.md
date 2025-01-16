# Задача 1: реализовать алгоритм, создающий из 2 изображений панораму. 

## Используемый метод
Для создания панорамы используется метод `cv.Stitcher.create(cv.Stitcher_PANORAMA)` из библиотеки OpenCV. Этот метод автоматически объединяет два изображения в одно панорамное изображение.

## Установка

### Предварительные условия
- Python 3.x
- OpenCV
- Matplotlib

### Установка зависимостей
Установите необходимые библиотеки с помощью следующей команды:
```bash
pip install numpy opencv-python matplotlib
```

## Использование

### Пример кода
```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def Panorama(path1='pan1.jpg', path2='pan2.jpg'):
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    images = [img1, img2]
    stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)
    if status == cv.Stitcher_OK:
        cv.imwrite('Panorama.jpg', panorama)
        cv.imshow('Panorama', panorama)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Error during stitching:", status)

Panorama()
```

### Шаги для использования
1. Убедитесь, что у вас установлены все необходимые библиотеки.
2. Поместите два изображения, которые вы хотите объединить в панораму, в ту же директорию, что и скрипт. Назовите их `pan1.jpg` и `pan2.jpg`.
3. Запустите скрипт:
   ```bash
   python panorama.py
   ```
4. Результат будет сохранен в файл `Panorama.jpg` и отображен в окне.

## Пример использования
1. Поместите два изображения `pan1.jpg` и `pan2.jpg` в ту же директорию, что и скрипт.
2. Запустите скрипт.
3. Результат будет сохранен в файл `Panorama.jpg` и отображен в окне.

![Ключевые точки](https://github.com/ann04ka/Homography_cvlab/blob/master/pankeypoints.jpg)

![Панорама](https://github.com/ann04ka/Homography_cvlab/blob/master/panorama.jpg)

# Задача 2: реализовать алгоритм поиска совпадений между 2 изображениями и выделения объекта интереса на сцене. 
## Используемый метод
Для нахождения соответствий между изображениями используется следующий метод:
1. **SIFT**: Алгоритм SIFT используется для обнаружения и описания ключевых точек в изображениях.
2. **FLANN**: Метод FLANN используется для поиска ближайших соседей и соответствий между дескрипторами ключевых точек.
3. **RANSAC**: Метод RANSAC используется для нахождения гомографии между двумя изображениями на основе найденных соответствий.

## Установка

### Предварительные условия
- Python 3.x
- OpenCV
- Matplotlib

### Установка зависимостей
Установите необходимые библиотеки с помощью следующей команды:
```bash
pip install numpy opencv-python matplotlib
```

## Использование

### Пример кода
```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def FindMatches(path1='photo_1.jpg', path2='photo_2.jpg', k=0.5):
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
    MIN_MATCH_COUNT = 10
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < k*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()

FindMatches(k=0.9)
```

### Шаги для использования
1. Убедитесь, что у вас установлены все необходимые библиотеки.
2. Поместите два изображения, для которых вы хотите найти соответствия, в ту же директорию, что и скрипт. Назовите их `photo_1.jpg` и `photo_2.jpg`.
3. Запустите скрипт:
   ```bash
   python find_matches.py
   ```
4. Результат будет отображен в окне с использованием Matplotlib.

## Пример использования
1. Поместите два изображения `photo_1.jpg` и `photo_2.jpg` в ту же директорию, что и скрипт.
2. Запустите скрипт.
3. Результат будет отображен в окне с использованием Matplotlib.
   
![Панорама](https://github.com/ann04ka/Homography_cvlab/blob/master/Homography.jpg)
