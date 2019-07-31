from run import *

if __name__ == '__main__':
    face_detection_model, face_validation_model, speaker_validation = load_models()
    video_dir = r'C:/Users/haoli/Desktop/video/蔡依林/song-1.mp4'
    process_single_video(video_dir, video_dir, face_detection_model, face_validation_model, speaker_validation)
