import threading
import cv2
import numpy as np
import time
import game
import detection_game_ver1

# from matplotlib import pyplot as plt

def cameras_function(lock_hit,lock_pop,event_start):
    # Global variabels
    global frame_hit_list
    global frame_pop_list
    global count_pop
    global N
    global vid_hit
    global vid_pop
    global oldest_frame
    # Init variabels
    frame_hit_list = [[np.ndarray(shape=(480,640,3),dtype='uint8') for i in range(N)] for j in range(6) ]
    frame_pop_list = [[np.ndarray(shape=(480,640,3),dtype='uint8') for i in range(N)] for j in range(6) ]
    oldest_frame = 0
    vid_hit = cv2.VideoCapture(1,cv2.CAP_DSHOW)#CAP_MSMF, CAP_DSHOW
    vid_pop = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    vid_hit.set(cv2.CAP_PROP_FPS, 30)
    vid_pop.set(cv2.CAP_PROP_FPS, 30)

    print("start play!!!")
    event_start.set()
    while(True):
        _, frame_hit_curr = vid_hit.read()
        _, frame_pop_curr = vid_pop.read()
        if count_pop > 5:
            break
        else:
            if np.any((frame_hit_curr[:,:,2] > 170) * (70 < frame_hit_curr[:,:,1]) *( frame_hit_curr[:,:,1]<200) * (frame_hit_curr[:,:,0] < 80)):
                curr_pop = count_pop
                lock_hit[oldest_frame].acquire()
                lock_pop[oldest_frame].acquire()
                # cv2.imwrite(str(count_pop) + "_"+str(oldest_frame)+'hit_curr.jpg',frame_pop_curr)
                frame_hit_list[curr_pop][oldest_frame] = frame_hit_curr
                frame_pop_list[curr_pop][oldest_frame] = frame_pop_curr

                lock_hit[oldest_frame].release()
                lock_pop[oldest_frame].release()
                oldest_frame = (oldest_frame + 1) % N


def detect_direction_change(lock_pop,lock_hit,game_detection,threshold=0):
    """computes the dot product between the previous flow and the current 
    flow and checks if the dot product is negative and exceeds the threshold. 
    If it satisfies both conditions, it indicates a change in direction,
    and the function returns True. Otherwise, it returns False.

    Args:
        frames (List[np.ndarray]): A list of video frames as NumPy arrays.
        threshold: A float representing the threshold for detecting change of direction.
    Returns:
        A boolean indicating whether there was a change of direction or not.
    """
    global frame_pop_list
    global frame_hit_list
    global oldest_frame
    global count_pop
    global N
    global frame

    # Initialize the frame and convert to gray scale
    
    basePointer = 0
    frame = [np.ndarray(shape=(480,640,3),dtype='uint8') for i in range(4)]
    pop_frame = [np.ndarray(shape=(480,640,3),dtype='uint8') for i in range(4)]

    while(True):
        badFrame = False
        for i in range(4):
            lock_hit[(basePointer+i)%N].acquire()
            lock_pop[(basePointer+i)%N].acquire()
        if count_pop < 6:
            for i in range(4):
                frame[i] = cv2.cvtColor(frame_hit_list[count_pop][(basePointer+i)%N], cv2.COLOR_BGR2GRAY)
               
                if np.max(frame[i][40:,:])==0:
               
                    for j in range(4):
                        lock_hit[(basePointer+j)%N].release()
                        lock_pop[(basePointer+j)%N].release()
                    basePointer = (basePointer + 1) % N
                    badFrame = True
                    break
                else:  
                    pop_frame[i] = frame_pop_list[count_pop][(basePointer+i)%N]
                    cv2.imwrite(str(count_pop) + 'hit'+str(i)+'.jpg',frame[i])
        if badFrame:
            continue
        for i in range(4):
            lock_hit[(basePointer+i)%N].release()
            lock_pop[(basePointer+i)%N].release()
    
        #compute the optical flow of previous and corrent frames
        prev_flow = cv2.calcOpticalFlowFarneback(frame[0], frame[1], None, 0.5, 3, 20, 3, 7, 1.5, 0)
        flow = cv2.calcOpticalFlowFarneback(frame[2], frame[3], None, 0.5, 3, 20, 3, 7, 1.5, 0)
        #compute dot product of 3d arrays
        dot_product = np.sum(flow * prev_flow, axis=-1)
        if dot_product.mean() < 0 and np.abs(dot_product.mean()) > threshold:
            time.sleep(0.5)
            hit_flag = False
            if count_pop < 6:
                x = 0
                y = 0
                #change to  2 from 4
                for j in range(2):
                    cv2.imwrite(str(count_pop) +'Noams_pop'+str(j)+'.jpg',pop_frame[j+1])
                    hit_flag_curr = detection_game_ver1.hit_miss(pop_frame[j + 1])
                    hit_flag = hit_flag or hit_flag_curr
                if hit_flag:
                    event_hit.set()
                # time.sleep(1)
                if count_pop < 6:
                    count_pop = count_pop + 1
                    print(str(count_pop) + "count_pop")
                    game_detection.set()
                else:
                    break

        basePointer = (basePointer + 1) % N

def game_main(event_hit,game_detection,event_start):
    event_start.wait()
    game.menu(event_hit, game_detection)
    event_start.clear()

if __name__ =="__main__":
    global N
    global count_pop

    count_pop = 0
    N = 16
    lock_hit = [threading.Lock() for j in range(N)]
    lock_pop = [threading.Lock() for j in range(N)]
    lock_frame = threading.Lock()

    game_detection = threading.Event()
    event_start = threading.Event()
    event_hit = threading.Event()

    test_T = threading.Thread(target=cameras_function, args=(lock_hit,lock_pop,event_start, ))
    t3 = threading.Thread(target=detect_direction_change, args=(lock_pop,lock_hit,game_detection,0.1, ))

    test_T.start()
    t3.start()

    game_main(event_hit, game_detection, event_start)

    test_T.join()
    t3.join()