import cv2, os, tempfile
from pathlib import Path
import numpy as np

#assuming there is a running openface container called "openface"
#assuming $TMP has been mounted to the openface container
TMP='/tmp' #'/dev/shm'
SAMPLE_INTERVAL=0.1
MAX_COUNT=20

def video2frame(video_path, interval=SAMPLE_INTERVAL, tstart=0, max_count=MAX_COUNT, save_to=''):
    print(video_path)
    if (not os.path.isdir(save_to)):
        os.mkdir(save_to)
    name = Path(video_path).stem   
    c = 0
    t = tstart
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_MSEC, t)
    success,image = video_capture.read()
    while success:
        cv2.imwrite(os.path.join(save_to,"%s_frame%d_timestamp%.3f.jpg" %(name,c,t)), image)     # save frame as JPEG file 
        print('New frame: %s_frame%d.jpg'%(name,c), success)
        c += 1
        t += interval
        video_capture.set(cv2.CAP_PROP_POS_MSEC,t*1000)
        success,image = video_capture.read()
        if c >= max_count:
            break
            
def docker_openface_wrapper(frame_dir, save_to, container_id='openface'):
    name = frame_dir.split("/")[-1]
    os.system("docker cp %s/ %s:/home/openface-build/"%(frame_dir, container_id))
    os.system("docker exec %s /home/openface-build/build/bin/FaceLandmarkImg -fdir /home/openface-build/%s -out_dir /home/openface-build/%s_openfaced"%(container_id,name,name))
    os.system("docker cp %s:/home/openface-build/%s_openfaced %s"%(container_id,name,save_to))
    
def process_openface_output(result_dir):
    ret = []
    for r in os.listdir(result_dir):
        if r.endswith(".csv"):
            n_frame = int(r.split('frame')[-1].split("_")[0])
            timestamp = float(r.split('timestamp')[-1].split(".csv")[0])
            ret.append(np.concatenate((np.array([n_frame,timestamp]), np.genfromtxt(os.path.join(result_dir,r), delimiter=',')[1])))
                
    return np.array(ret)    

def openface_a_video(video_path, container_id='openface'):
    with tempfile.TemporaryDirectory(dir=TMP) as tpdir:
        video2frame(video_path, 0.1, save_to=tpdir)
        result_dir = os.path.join(tpdir,'openfaced')
        os.system("docker exec %s /home/openface-build/build/bin/FaceLandmarkImg -fdir %s -out_dir %s"%(container_id,tpdir,result_dir))
        ret = process_openface_output(result_dir)
        #files created by container cannot be deleted by python file manager, so delete them by docker exec
        os.system("docker exec %s rm -r %s"%(container_id, result_dir))         
        
    return ret

