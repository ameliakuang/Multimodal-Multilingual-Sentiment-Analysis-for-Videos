from visual import *
import os, pathlib
from multiprocessing import Process

video_list = []
videos_dir = '/media/dominic/LEXAR/Video'
for t in os.listdir(videos_dir):
    if os.path.isdir(os.path.join(videos_dir,t)):
        for m in os.listdir(os.path.join(videos_dir,t)):
            if m.endswith('.mp4'):
                video_list.append(os.path.join(videos_dir,t,m))

print(len(video_list))          
#merge_videos(video_list, container_id='openface1', save_to='/home/dominic/11777_project/french_amt_b1_b2_visual_features_p1.npz')

if __name__=='__main__':
     p1 = Process(target = merge_videos, args=(video_list[:1000], 'openface1', None, '/home/dominic/11777_project/french_amt_b1_b2_visual_features_p1.npz'))
     p1.start()
     p2 = Process(target = merge_videos, args=(video_list[1000:2000], 'openface2', None, '/home/dominic/11777_project/french_amt_b1_b2_visual_features_p2.npz'))
     p2.start()
     p3 = Process(target = merge_videos, args=(video_list[2000:3000], 'openface3', None, '/home/dominic/11777_project/french_amt_b1_b2_visual_features_p3.npz'))
     p3.start()
     p4 = Process(target = merge_videos, args=(video_list[3000:4000], 'openface4', None, '/home/dominic/11777_project/french_amt_b1_b2_visual_features_p4.npz'))
     p4.start()
     p5 = Process(target = merge_videos, args=(video_list[4000:], 'openface5', None, '/home/dominic/11777_project/french_amt_b1_b2_visual_features_p5.npz'))
     p5.start()
