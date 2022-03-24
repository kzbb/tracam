import cv2
import numpy
import mediapipe
import time
import pyvirtualcam


class FaceDetector:
    def __init__(self, el_max=60):
        self.img = None
        self.face_detection = mediapipe.solutions.face_detection.FaceDetection(model_selection=0,
                                                                               min_detection_confidence=0.5)
        self.drawing = mediapipe.solutions.drawing_utils

        # 移動平均をとるための入れ物たち。
        # 配列に格納する要素数
        self.el_max = el_max
        # 横幅（拡大率の変化）は座標移動よりもさらに緩やかになるように要素数を増やす
        self.el_max_w = self.el_max * 3
        # 切り出し領域の中心
        self.arr_x = []
        self.arr_y = []
        # はじめは引きの画からにしたいので、幅については１で初期化
        self.arr_w = []
        # self.arr_w = [1.0] * self.el_max_w

    def detect_face(self, img, draw=True):
        # まずは画像を読み込み
        self.img = img
        # 顔検出のため色空間を変換
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 顔検出
        results = self.face_detection.process(self.img)
        # 色空間をもとに戻す
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        # 顔の位置を記録するための入れ物たち
        center_x = []
        center_y = []
        left_side_of_faces = []
        right_side_of_faces = []

        if results.detections:
            # 顔が検出されたら、顔ひとつごとに位置を記録
            for detection in results.detections:
                # 鼻の座標を配列に入れておく
                # relative_keypoints
                # 0:right eye, 1:left eye, 2:nose tip, 3:mouth center, 4:right ear tragion, 5:left ear tragion
                center_x.append(detection.location_data.relative_keypoints[2].x)
                center_y.append(detection.location_data.relative_keypoints[2].y)

                # 顔の左右を配列に入れておく
                left_side_of_faces.append(detection.location_data.relative_bounding_box.xmin)
                right_side_of_faces.append(
                    detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width)

                # 顔の検出状況を描画
                if draw:
                    self.drawing.draw_detection(self.img, detection)

            # 顔は複数あるかもしれない。それぞれの鼻の位置の平均をとって切り出し領域の中心とする
            roi_center_x = numpy.average(center_x)
            roi_center_y = numpy.average(center_y)
            # 顔は複数あるかも知れない。全ての顔の中で左側最小値と右側最大値をとって切り出し領域の横幅とする
            roi_width = max(right_side_of_faces) - min(left_side_of_faces)

        else:
            # 顔が検出されなかった場合はズームアウト
            roi_center_x = 0.5
            roi_center_y = 0.5
            roi_width = 1.0

        # 以下、座標、切り出し幅、それぞれ配列に格納しておく

        # 切り出し領域の中心（ｘ座標）
        self.arr_x.append(roi_center_x)
        if len(self.arr_x) > self.el_max:
            self.arr_x.pop(0)

        # 切り出し領域の中心（ｙ座標）
        self.arr_y.append(roi_center_y)
        if len(self.arr_y) > self.el_max:
            self.arr_y.pop(0)

        # 切り出し領域の横幅
        self.arr_w.append(roi_width)
        if len(self.arr_w) > self.el_max_w:
            self.arr_w.pop(0)

    def crop_image(self, img, margin=4.5):
        # まずは画像をセット
        self.img = img

        # 切り出し領域の座標と幅について、配列から移動平均をとる
        roi_center_x_average = numpy.average(self.arr_x)
        roi_center_y_average = numpy.average(self.arr_y)
        roi_width_average = numpy.average(self.arr_w)

        # 切り出し領域の横幅に余白を加える。余白を含めて最大値は１
        roi_width_average = min([roi_width_average * margin, 1.0])

        # 幅は画像横幅に対する割合を小数で表したものになっている。そのまま高さにも適用する
        roi_height = roi_width_average

        # 切り出し座標に代入
        x1 = roi_center_x_average - roi_width_average / 2
        x2 = roi_center_x_average + roi_width_average / 2
        y1 = roi_center_y_average - roi_height / 2
        y2 = roi_center_y_average + roi_height / 2

        # 安全装置。切り出し領域が画像からはみ出ないようにする
        if x1 < 0.0:
            x1 = 0.0
            x2 = roi_width_average

        if x2 > 1.0:
            x2 = 1.0
            x1 = 1.0 - roi_width_average

        if y1 < 0.0:
            y1 = 0.0
            y2 = roi_height

        if y2 > 1.0:
            y2 = 1.0
            y1 = 1.0 - roi_height

        # 確認用の矩形描画
        # cv2.rectangle(self.img, pt1=(int(x1*iw), int(y1*ih)),pt2=(int(x2*iw),int(y2*ih)),color=(0,255,0),thickness=1)

        # 切り出しとリサイズ
        ih, iw, ic = self.img.shape
        self.img = self.img[int(ih * y1):int(ih * y2), int(iw * x1):int(iw * x2)]
        self.img = cv2.resize(self.img, (iw, ih))

        return self.img


class Bokeh:
    def __init__(self, el_max=90):
        self.img = None
        self.bg_img = None
        self.selfie_segmentation = mediapipe.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.drawing = mediapipe.solutions.drawing_utils

    def bokeh(self, img, threshold=0.6, coc=50):
        # まずは画像を読み込み
        self.img = img
        # 前景検出のため色空間を変換
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 前景検出
        results = self.selfie_segmentation.process(self.img)
        # 色空間をもとに戻す
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        # 背景画像を作成
        coc_pix = int(self.img.shape[0] / coc)
        self.bg_img = cv2.blur(self.img, (coc_pix, coc_pix))
        # self.bg_img = cv2.resize(cv2.resize(self.img, (self.img.shape[1] // 10, self.img.shape[0] // 10)), (self.img.shape[1], self.img.shape[0]))

        # 合成の準備
        condition = numpy.stack((results.segmentation_mask,) * 3, axis=-1) > threshold
        # 合成
        output_image = numpy.where(condition, self.img, self.bg_img)

        return output_image


class DisplayInformation:
    def __init__(self):
        self.img = None
        self.p_time = 0
        self.c_time = 0

    def set_image(self, img):
        self.img = img

    def display_all_info(self, img):
        self.set_image(img)
        self.fps()
        return self.img

    def fps(self):
        # フレームレートを計って表示
        self.c_time = time.time()
        fps = 1 / (self.c_time - self.p_time)
        self.p_time = self.c_time
        cv2.putText(self.img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


def main():
    # 内蔵カメラの番号、基本的には0番だが、
    # OBSをインストールしていると、0番がOBSのバーチャルカメラになっており、
    # 内蔵カメラが1番になっていることもある。
    cap = cv2.VideoCapture(0)

    # 顔検出器
    face_detector = FaceDetector()
    # 背景ぼかし
    bokeh = Bokeh()
    # 画面上に情報を表示
    info = DisplayInformation()

    # OBSがインストールされていれば、バーチャルカメラに映像を送ることもできる。
    # バーチャルカメラ使用時にOBSが起動している必要はない。インストールされていればよい。
    as_virtual_camera = True

    # バーチャルカメラを使う場合の処理
    if as_virtual_camera:
        # 最初に１枚、バーチャルカメラのために画像を取得
        res, img = cap.read()
        cam = pyvirtualcam.Camera(width=img.shape[1], height=img.shape[0], fps=24)

    while True:
        # カメラから画像読み込み
        res, img = cap.read()

        # 顔検出
        face_detector.detect_face(img, draw=False)
        # 顔周辺を切り出し
        img = face_detector.crop_image(img)

        # 背景ぼかし
        img = bokeh.bokeh(img)

        # 情報表示
        # img = info.display_all_info(img)

        # バーチャルカメラとして使う場合、映像をカメラに送る
        if as_virtual_camera:
            cam.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # cam.sleep_until_next_frame()

        # 映像を表示
        cv2.imshow('Viewer', img)

        # ESCキーで終了
        key = cv2.waitKey(1)
        if key == 27:
            break

    # 終了処理
    cap.release()
    if as_virtual_camera:
        cam.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
