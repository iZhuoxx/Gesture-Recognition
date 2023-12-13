import mediapipe as mp
import time
import paddle
import cv2
import webbrowser
from paddle.vision import transforms
import numpy as np
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def cmpx(a):
    return a[0]


def cmpy(a):
    return a[1]


hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=9,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(1)
fps_video = cap.get(cv2.CAP_PROP_FPS)

# 导入模型
net = paddle.jit.load("./model/ResNet/ResNet")
i = 0
time1 = 0
num_true = 7
time0 = 0
hand_num = 0
wait = []
last_num = 0
print("保持0手势1s以上中即可解锁")
x1_final = 0
x2_final = 0
y1_final = 0
y2_final = 0
fps = 0
x1 = 0
x2 = 0
y1 = 0
y2 = 0
while True:

    shape = []
    lock = 1
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.flip(frame, 1)

    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame2 = frame
    start = time.time()

    # if results.multi_handedness:
    #     for hand_label in results.multi_handedness:
    #         print(hand_label)
    #         pass
    if results.multi_hand_landmarks:
        i += 1
        hand_landmarks = results.multi_hand_landmarks[0]
        # for hand_landmarks in results.multi_hand_landmarks:
        if i == 3:
            i = 0
            for name, hand in enumerate(hand_landmarks.landmark):
                shape.append([hand.x, hand.y])
                # print('hand_landmarks:', hand.x)
            shape.sort(key=cmpx)
            x1 = int(max(shape[0][0] * 640, 0))  # -50,0))
            x2 = int(min(shape[20][0] * 640, 640))  # +50,640))
            shape.sort(key=cmpy)
            y1 = int(max(shape[0][1] * 480, 0))  # - 50, 0))
            y2 = int(min(shape[20][1] * 480, 480))  # + 50, 480))
            # print(x1,x2,y1,y2)
            x_half = (y2 - y1) / 2.5
            y_half = (x2 - x1) / 2.5
            frame = frame[:, :, [2, 1, 0]]
            # frame = cv2.flip(frame, 1)
            frame = Image.fromarray(np.uint8(np.array(frame)))
            # frame.show()
            x1_final = max(x1 - x_half, 0)
            x2_final = min(640, x2 + x_half)
            y1_final = max(y1 - y_half, 0)
            y2_final = min(y2 + y_half, 480)
            # frame = frame.crop((max(x1 - x_half, 0), max(y1 - y_half, 0), min(640, x2 + x_half), min(y2 + y_half, 480)))
            frame = frame.crop((x1_final, y1_final, x2_final, y2_final))
            # frame = frame.crop((x1, y1, x2, y2))
            # frame.show()
            frame = frame.resize((120, 120), Image.BICUBIC)
            # frame = cv2.fastNlMeansDenoisingColored(frame2, None, 5, 5, 7, 21)
            # frame.show()
            img = transforms.ToTensor()(frame).unsqueeze(0)

            num0 = net(img)
            end = time.time()
            seconds = end - start  
            fps = 1 / seconds  
            fps = "%.2f fps" % fps
            # num = np.argmax(net(img))
            num_true = np.argmax(num0)
            num0 = num0.tolist()
            probability = num0[0][num]
            if probability > 0.8:
                print("\r数字是:{0}  "
                      "解锁后输出手势是:{1}".format(num_true, hand_num), end="")
                # time2 = time.localtime().tm_sec
                time2 = time.time()
                if num_true == 0:
                    # time1 = time.localtime().tm_sec
                    time0 = time.time()
                    if len(wait) == 0:
                        wait.append(time0)
                    if wait[0] + 1 < time0:
                        time1 = time.time()
                # print("\n\rtime1:{0}, time2:{1}".format(time1, time2), end="")
                if time1 + 1 > time2 > time1 + 0.5 and num_true != 0:
                    hand_num = num_true
                    lock = 0
                    print("\r数字是:{0}  "
                          "解锁后输出手势是:{1}".format(num_true, hand_num), end="")
                if num_true != 0:
                    wait = []


        mp_drawing.draw_landmarks(
            frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if num_true != 7:
        cv2.putText(frame2, "Gesture is: {0}".format(num_true), (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # more = 50

    cv2.rectangle(frame2, (int(x2_final), int(y2_final)), (int(x1_final), int(y1_final)), (255, 0, 0), 2)
    # cv2.rectangle(frame2, (int(x2+more), int(y2+more)), (int(x1-more), int(y1-more)), (0, 0, 255), 2)
    # print("x1 is :{0}".format(x1))
    # print("x2 is :{0}".format(x2))

    # cv2.putText(frame2, str(fps), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow('MediaPipe Hands', frame2)  # [int(y1):int(y2), int(x1):int(y2)])

    if (num_true == 1) & (lock == 0):
        print("\n1")
        webbrowser.open("https://www.google.com/", new=1, autoraise=True)
        time.sleep(1)  # 休眠1秒
    last_num = num_true


    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
