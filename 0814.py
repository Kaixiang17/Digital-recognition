import cv2
import numpy as np

#數字特徵 (預定義)
known_features = {
    0: [15455, 117.67, 70.05, 13.81, 0.75],
    1: [6144, 112.57, 227.40, 35.34, 0.42],
    2: [9790, 118.45, 516.69, 51.78, 0.40],
    3: [9252, 116.28, 364.15, 48.37, 0.37],
    4: [15020, 434.44, 512.26, 14.90, 0.72],
    5: [13413, 439.60, 216.28, 27.65, 0.66],
    6: [13417, 427.77, 660.88, 27.37, 0.66],
    7: [10896, 121.82, 664.79, 23.50, 0.72],
    8: [7037, 416.63, 368.54, 41.28, 0.36],
    9: [10508, 434.28, 70.06, 57.60, 0.49]
}


def shape_feature(f, method):
    if method == 1:
        return np.count_nonzero(f)
    elif method == 2:
        moments = cv2.moments(f, binaryImage=True)
        if moments['m00'] != 0:
            xc = moments['m10'] / moments['m00']
            yc = moments['m01'] / moments['m00']
        else:
            xc = yc = 0
        return xc, yc
    elif method == 3:  #
        area = shape_feature(f, 1)
        contours, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        return (perimeter * perimeter) / area if area > 0 else 0
    elif method == 4:
        area = shape_feature(f, 1)
        contours, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, radius) = cv2.minEnclosingCircle(contours[0]) if contours else (0, 1)
        circular_area = np.pi * radius ** 2
        return area / circular_area if circular_area > 0 else 0
    return 0


def preprocess_image(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((5, 5), np.uint8)


    opened_image = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    return img, closed_image


def match_digit(features, known_features):
    min_distance = float('inf')
    best_match = None
    for digit, k_features in known_features.items():

        feature_vector = np.array([features[0], features[1][0], features[1][1], features[2], features[3]])
        k_feature_vector = np.array(k_features)

        # 歐式距離
        distance = np.linalg.norm(feature_vector - k_feature_vector)

        if distance < min_distance:
            min_distance = distance
            best_match = digit
    return best_match


def main():
    img_path = 'C:\\myOpenCV\\images\\number_white.jpg'
    img, processed_image = preprocess_image(img_path)


    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # （紅色）
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


        mask = np.zeros_like(processed_image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)


        area = shape_feature(mask, 1)
        xc, yc = shape_feature(mask, 2)
        compactness = shape_feature(mask, 3)
        circularity = shape_feature(mask, 4)

        # 匹配已知數字
        features = [area, (xc, yc), compactness, circularity]
        digit = match_digit(features, known_features)


        cv2.circle(img, (int(xc), int(yc)), 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.putText(img, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        print(f"Contour {i}:")
        print(f"  Area: {area}")
        print(f"  Geometric Center: ({xc:.2f}, {yc:.2f})")
        print(f"  Compactness: {compactness:.2f}")
        print(f"  Circularity: {circularity:.2f}")
        print(f"  Recognized Digit: {digit}")

    cv2.imshow("Detected Digits", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
