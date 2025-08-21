class Const:
    """
    Constants used throughout the facial feature analysis project.
    Contains keys for statistics, DeepFace outputs, and possible values for categorical features.
    """
    # Access keys for statistics
    total_faces = 'total_faces'  # Total number of faces detected
    total_faces_mixed = 'total_faces_mixed'  # Total faces in mixed images
    total_imgs = 'total_imgs'  # Total number of images
    bad_imgs = 'bad_imgs'  # Number of bad images
    mixed_imgs = 'mixed_imgs'  # Number of images with mixed attributes
    mixed_gender = 'mixed_gender'  # Images with mixed gender
    mixed_race = 'mixed_race'  # Images with mixed race
    mixed_emotion = 'mixed_emotion'  # Images with mixed emotion
    mixed_ages = 'mixed_ages'  # Images with mixed ages
    num_faces = 'num_faces'  # Number of faces per image
    total_apparences = 'total_apparences'  # Total appearances
    total_apparences_mixed = 'total_apparences_mixed'  # Total appearances in mixed images
    avg_depth = 'avg_depth'  # Average depth of faces
    face_position_per = 'face_position_per'  # Face position as percentage
    face_position_norm = 'face_position_norm'  # Normalized face position
    gender = 'gender'  # Gender attribute
    age = 'age'  # Age attribute
    age_range = 'age_range'  # Age range attribute
    emotion = 'emotion'  # Emotion attribute
    centrality = 'centrality'  # Centrality of face in image
    face_center_x = 'face_center_x'  # X coordinate of face center
    face_center_y = 'face_center_y'  # Y coordinate of face center
    race = 'race'  # Race attribute
    in_front_of_count = 'in_front_of_count'  # Count of faces in front
    in_front_of_val = 'in_front_of_val'  # Value for faces in front
    group = 'group'  # Group attribute
    race_nodes = 'race_nodes'  # Race nodes for graph analysis
    gender_nodes = 'gender_nodes'  # Gender nodes for graph analysis
    combined = 'combined'  # Combined key for data
    all_key = -1  # Key for all data
    x = 'x'  # X coordinate
    y = 'y'  # Y coordinate
    w = 'w'  # Width
    h = 'h'  # Height
    max_depth = 'max_depth'  # Maximum depth
    mean_depth = 'mean_depth'  # Mean depth

    # Access keys for DeepFace outputs
    dominant_emotion = 'dominant_emotion'  # Most likely emotion
    region =  'region'  # Region of face
    dominant_gender = 'dominant_gender'  # Most likely gender
    dominant_race = 'dominant_race'  # Most likely race

    # Possible values for categorical features
    __possible_emotions = ["happy", "fear", "neutral", "sad", "angry", "surprise", "disgust"]  # List of possible emotions
    __possible_age = list(range(0, 110))  # List of possible ages (0-109)
    __possible_races = ["middle eastern", "white", "indian", "asian", "black", "latino hispanic"]  # List of possible races
    __possible_genders = ["Man", "Woman"]  # List of possible genders

    # Dictionary mapping feature names to their possible values
    possible_values = {
        age: __possible_age,
        emotion: __possible_emotions,
        gender: __possible_genders,
        race: __possible_races
    }
