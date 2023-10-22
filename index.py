import cv2 as cv

def load_image(file_path):
    return cv.imread(file_path)

def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)
    return cv.GaussianBlur(clahe_result, (5, 5), 100)

def extract_minutiae(image, original_image):
    sobely = cv.Sobel(src=image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
    sobely = cv.convertScaleAbs(sobely)
    _, binary_image = cv.threshold(sobely, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    minutiae = []
    color = (255, 0, 0) 

    for contour in contours:
        for i in range(len(contour)):
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            if len(approx) == 2:
                minutiae.append(approx[0])
                
                x, y = approx[0][0]
                cv.circle(original_image, (y, x), 10,color, 2)
                
            elif len(approx) > 2:
                minutiae.append(approx[0])
                

    return minutiae

def main():
    file_path = "test4.jpg"
    image = load_image(file_path)
    preprocessed_image = preprocess_image(image)
    sobely = cv.Sobel(src=preprocessed_image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)


    extract_minutiae(preprocessed_image, sobely)

    cv.imshow('Fingerprint', sobely)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()