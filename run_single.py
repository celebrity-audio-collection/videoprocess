from run import *

if __name__ == '__main__':
    face_detection_model, face_validation_model, speaker_validation = load_models()
    POI_imgs = [os.path.join('images/蔡依林', pic) for pic in
                os.listdir('images/蔡依林')]
    face_validation_model.update_POI(POI_imgs)
    video_dir = r'C:\Users\haoli\Desktop\video\蔡依林\song\song-1.mp4'
    process_single_video(video_dir, r'C:\Users\haoli\Desktop\video\蔡依林\song\song-1.txt',
                         face_detection_model, face_validation_model, speaker_validation,
                         r'C:\Users\haoli\Desktop\video\蔡依林\song')
