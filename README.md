## <div align="center">Pictuer Swap</div>

This app will run Yolov8 Image Segmentation model, trained on a custom dataset, to segment pictures hanging on a wall. Once the frame on the wall is detected another picture can be selected to swap with the original picture. 

After running an inference:
`results = model.predict(input_image, imgsz=640, conf=conf, save=False, device='cpu')`

`results` are parsed to extract boxes and masks. The coordinates then are used to capture the perspective and angles of a frame on the wall. 

```
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[-1]
    cv2.drawContours(canvas, cnt, -1, (255, 255, 0), 2)
```

The extracted coordinates then adjust the new picture. 
```
    pts2 = np.float32(new_corners)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows), bg_img, borderMode=cv2.BORDER_TRANSPARENT)
```
Finally, a merge function with overlay the new picture on the original frame. 
```
    for c in range(3):
        image[y1:y2, x1:x2, c] = alpha * img_pic[:, :, c] 
```


