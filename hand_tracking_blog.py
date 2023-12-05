import cv2
import mediapipe as mp


# Reference
# https://www.makeuseof.com/python-hand-tracking-opencv/
# Index finger tip is landmark 8
# Index finger base is landmark 5

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Returning the detected hands to calling function
    return results


# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for ind, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates on the terminal
                print(ind, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        return img


def main():
    # Replace 0 with the video path to use a  pre-recorded video
    cap = cv2.VideoCapture('data/participants/P17/A1/P17_A1.mp4')

    while True:
        # Taking the input
        success, image = cap.read()
        # image = imutils.resize(image, width=500, height=500)
        results = process_image(image)
        draw_hand_connections(image, results)

        # Displaying the output
        cv2.imshow("Hand tracker", image)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
