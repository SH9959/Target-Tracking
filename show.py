import cv2
frame_num = 1
pos = {}
name = f"f{frame_num}.png"
pic = cv2.imread(name)
LAST_FRAME = 329
def end():
    print(pos)
    exit(0)
    pass

def update():
    global frame_num ,name,pic
    frame_num = frame_num + 1
    if frame_num > LAST_FRAME:
        end()
        pass
    name = f"f{frame_num}.png"
    pic = cv2.imread(name)
    cv2.imshow(WINDOW,pic)
    pass

def mouse_cb(event:int,x:int,y:int,flags,param):
    global frame_num ,pos,frame_num,name
    if event == cv2.EVENT_LBUTTONDOWN:
        pos[frame_num] = (x,y)
        print(f"pos[{frame_num}] = {(x,y)}")
        update()
        return
    elif event == cv2.EVENT_RBUTTONDOWN:
        pos[frame_num] = None
        print(f"pos[{frame_num}] = {None}")
        update()
        return
    elif event == cv2.EVENT_MBUTTONDOWN:
        end()
    pass

WINDOW  = "frames"
cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WINDOW,mouse_cb)
cv2.imshow(WINDOW,pic)
cv2.waitKey(0)

