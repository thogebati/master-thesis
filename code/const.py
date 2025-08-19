class Const:
    # access key for statistics
    total_faces = 'total_faces'
    total_faces_mixed = 'total_faces_mixed'
    total_imgs = 'total_imgs'
    bad_imgs = 'bad_imgs'
    mixed_imgs = 'mixed_imgs'
    mixed_gender = 'mixed_gender'
    mixed_race = 'mixed_race'
    mixed_emotion = 'mixed_emotion'
    mixed_ages = 'mixed_ages'
    num_faces = 'num_faces'
    total_apparences = 'total_apparences'
    total_apparences_mixed = 'total_apparences_mixed'
    avg_depth = 'avg_depth'
    face_position_per = 'face_position_per'
    face_position_norm = 'face_position_norm'
    gender = 'gender'
    age = 'age'
    age_range = 'age_range'
    emotion = 'emotion'
    centrality = 'centrality'
    face_center_x = 'face_center_x'
    face_center_y = 'face_center_y'
    race = 'race'
    in_front_of_count = 'in_front_of_count'
    in_front_of_val = 'in_front_of_val'
    group = 'group'
    race_nodes = 'race_nodes'
    gender_nodes = 'gender_nodes'
    combined = 'combined'
    all_key = -1
    x = 'x'
    y = 'y'
    w = 'w'
    h = 'h'
    max_depth = 'max_depth'
    mean_depth = 'mean_depth'
    
    # access keys for DeepFace
    dominant_emotion = 'dominant_emotion'
    region =  'region'
    dominant_gender = 'dominant_gender'
    dominant_race = 'dominant_race'
    
    # possibles 
    __possible_emotions = ["happy", "fear", "neutral", "sad", "angry", "surprise", "disgust"]
    __possible_age = list(range(0, 110))
    __possible_races = ["middle eastern", "white", "indian", "asian", "black", "latino hispanic"]
    __possible_genders = ["Man", "Woman"]
    
    possible_values = {age: __possible_age, emotion: __possible_emotions, gender: __possible_genders, race: __possible_races}
    