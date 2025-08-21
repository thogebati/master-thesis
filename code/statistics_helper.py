import json
import numpy as np
import re
from const import Const

# Global node for storing all statistics
General_Node = {Const.num_faces: {}}

def read_pfm(path: str) -> tuple:
    """
    Read a PFM (Portable Float Map) file and return its data and scale.

    Args:
        path (str): Path to the PFM file.

    Returns:
        tuple: (data, scale), where data is a numpy array and scale is a float.
    """
    with open(path, "rb") as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        # Read image dimensions
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        # Read scale and determine endianness
        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            endian = "<"  # little-endian
            scale = -scale
        else:
            endian = ">"  # big-endian

        # Read image data
        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

class Helper:
    """
    Helper class for accumulating and organizing statistics for facial analysis.
    Stores results in the global General_Node dictionary.
    """

    def set_total_faces(self, total: int) -> None:
        """Set the total number of faces processed."""
        General_Node[Const.total_faces] = total

    def set_total_imgs(self, total: int) -> None:
        """Set the total number of images processed."""
        General_Node[Const.total_imgs] = total

    def set_bad_imgs(self, bad_imgs: int) -> None:
        """Set the number of images with errors or missing data."""
        General_Node[Const.bad_imgs] = bad_imgs

    def set_mixed_imgs(self, mixed_imgs: int) -> None:
        """Set the number of images with mixed demographic attributes."""
        General_Node[Const.mixed_imgs] = mixed_imgs

    def add_number_faces(self, num_faces: int) -> None:
        """Increment the count for a specific number of faces in an image."""
        faces_dict = self._get_num_faces()
        if str(num_faces) not in faces_dict:
            faces_dict[str(num_faces)] = 1
        else:
            faces_dict[str(num_faces)] += 1

    def _get_num_faces(self) -> dict:
        """Get or initialize the dictionary for counting number of faces."""
        if Const.num_faces not in General_Node:
            General_Node[Const.num_faces] = {}
        return General_Node[Const.num_faces]

    # =================== general helper ===================
    def _add_gender(self, gender_dict: dict, gender: str):
        """Increment gender count in a dictionary."""
        if gender not in gender_dict:
            for _gender in Const.possible_values[Const.gender]:
                gender_dict[_gender] = 0
        gender_dict[gender] = gender_dict[gender] + 1

    def _add_race(self, race_dict: dict, race: str):
        """Increment race count in a dictionary."""
        if race not in race_dict:
            for _race in Const.possible_values[Const.race]:
                race_dict[_race] = 0
        race_dict[race] = race_dict[race] + 1

    def _add_age(self, age_dict: dict, age: str):
        """Increment age count in a dictionary."""
        if age not in age_dict:
            for _age in Const.possible_values[Const.age]:
                age_dict[_age] = 0
        age_dict[age] = age_dict[age] + 1

    def _add_emotion(self, emotion_dict: dict, emotion: str):
        """Increment emotion count in a dictionary."""
        if emotion not in emotion_dict:
            for _emotion in Const.possible_values[Const.emotion]:
                emotion_dict[_emotion] = 0
        emotion_dict[emotion] = emotion_dict[emotion] + 1

    def _add_total_num_faces(self, _dict: dict):
        """Increment the total number of faces for a demographic group."""
        if Const.total_faces not in _dict:
            _dict[Const.total_faces] = 1
        else:
            _dict[Const.total_faces] = _dict[Const.total_faces] + 1

    def _add_total_num_faces_mixed(self, _dict: dict):
        """Increment the total number of faces for mixed demographic groups."""
        if Const.total_faces_mixed not in _dict:
            _dict[Const.total_faces_mixed] = 1
        else:
            _dict[Const.total_faces_mixed] = _dict[Const.total_faces_mixed] + 1

    def _add_total_apparences(self, _dict: dict):
        """Increment the total number of appearances for a demographic group."""
        if Const.total_apparences not in _dict:
            _dict[Const.total_apparences] = 1
        else:
            _dict[Const.total_apparences] += 1

    def _add_total_apparences_mixed(self, _dict: dict):
        """Increment the total number of appearances for mixed demographic groups."""
        if Const.total_apparences_mixed not in _dict:
            _dict[Const.total_apparences_mixed] = 1
        else:
            _dict[Const.total_apparences_mixed] += 1

    def _add_avg_depth(self, _dict: dict, avg_depth: int):
        """Add average depth value to the demographic group."""
        if Const.avg_depth not in _dict:
            _dict[Const.avg_depth] = [avg_depth]
        else:
            _dict[Const.avg_depth].append(avg_depth)

    def _add_first_position(self, _dict: dict, pos: str):
        """Increment the count for a specific face position."""
        if pos not in _dict:
            _dict[pos] = 1
        else:
            _dict[pos] += 1

    def _add_in_front_of_count(self, _dict: dict, value: str):
        """Increment the count for 'in front of' relationships."""
        if value not in _dict:
            _dict[value] = 1
        else:
            _dict[value] += 1

    def _add_in_front_of_val(self, _dict: dict, other_value: str, weighted_val: float):
        """Add a weighted value for 'in front of' relationships."""
        if other_value not in _dict:
            _dict[other_value] = {}
        _dict[other_value][str(len(_dict[other_value]))] = weighted_val

    def add_combined_info(self, _dict: dict):
        """Add a combined demographic info dictionary to the global node."""
        Top_note = General_Node[Const.num_faces]
        if Const.combined not in Top_note:
            Top_note[Const.combined] = []
        Top_note[Const.combined].append(_dict)

    # =================== nodes getter ===================
    def _get_gender_node(self, _dict: dict) -> dict:
        """Get or initialize the gender node in a dictionary."""
        if Const.gender not in _dict:
            _dict[Const.gender] = {}
        return _dict[Const.gender]

    def _get_age_node(self, _dict: dict) -> dict:
        """Get or initialize the age node in a dictionary."""
        if Const.age not in _dict:
            _dict[Const.age] = {}
        return _dict[Const.age]

    def _get_emotion_node(self, _dict: dict) -> dict:
        """Get or initialize the emotion node in a dictionary."""
        if Const.emotion not in _dict:
            _dict[Const.emotion] = {}
        return _dict[Const.emotion]

    def _get_race_node(self, _dict: dict) -> dict:
        """Get or initialize the race node in a dictionary."""
        if Const.race not in _dict:
            _dict[Const.race] = {}
        return _dict[Const.race]

    def _get_in_front_of_count_node(self, _dict: dict) -> dict:
        """Get or initialize the in_front_of_count node in a dictionary."""
        if Const.in_front_of_count not in _dict:
            _dict[Const.in_front_of_count] = {}
        return _dict[Const.in_front_of_count]

    def _get_in_front_of_val_node(self, _dict: dict) -> dict:
        """Get or initialize the in_front_of_val node in a dictionary."""
        if Const.in_front_of_val not in _dict:
            _dict[Const.in_front_of_val] = {}
        return _dict[Const.in_front_of_val]

    def _get_face_position_node(self, _dict: dict) -> dict:
        """Get or initialize the face_position_per node in a dictionary."""
        if Const.face_position_per not in _dict:
            _dict[Const.face_position_per] = {}
        return _dict[Const.face_position_per]

    # =================== num_faces helper ===================
    def _get_num_faces_top_node(self, num_faces: int):
        """
        Get the top node for a given number of faces.
        For groups with 5 or more faces, use the 'group' node.
        """
        if num_faces >= 5:
            if Const.group not in General_Node[Const.num_faces]:
                General_Node[Const.num_faces][Const.group] = {}
            return General_Node[Const.num_faces][Const.group]
        else:
            if str(num_faces) not in General_Node[Const.num_faces]:
                General_Node[Const.num_faces][str(num_faces)] = {}
            return General_Node[Const.num_faces][str(num_faces)]

    # =================== race helper ===================
    def _get_race_top_node(self, race: str):
        """Get or initialize the top node for a race."""
        Top_note = General_Node[Const.num_faces]
        if Const.race_nodes not in Top_note:
            Top_note[Const.race_nodes] = {}
        if race not in Top_note[Const.race_nodes]:
            Top_note[Const.race_nodes][race] = {}
        return Top_note[Const.race_nodes][race]

    def add_race_gender(self, race: str, gender: str):
        """Add gender statistics under a race node."""
        race_node = self._get_race_top_node(race)
        gender_node = self._get_gender_node(race_node)
        self._add_gender(gender_dict=gender_node, gender=gender)

    def add_race_age(self, race: str, age: int):
        """Add age statistics under a race node."""
        race_dict = self._get_race_top_node(race)
        age_dict = self._get_age_node(race_dict)
        self._add_age(age_dict=age_dict, age=age)

    def add_race_emotion(self, race: str, emotion: str):
        """Add emotion statistics under a race node."""
        race_node = self._get_race_top_node(race)
        emotion_node = self._get_emotion_node(race_node)
        self._add_emotion(emotion_dict=emotion_node, emotion=emotion)

    def add_race_total_num_faces(self, race: str):
        """Increment total number of faces for a race."""
        race_node = self._get_race_top_node(race)
        self._add_total_num_faces(race_node)

    def add_race_total_num_faces_mixed(self, race: str):
        """Increment total number of faces for mixed race groups."""
        race_node = self._get_race_top_node(race)
        self._add_total_num_faces_mixed(race_node)

    def add_race_total_apparences(self, race: str):
        """Increment total appearances for a race."""
        race_node = self._get_race_top_node(race)
        self._add_total_apparences(race_node)

    def add_race_total_apparences_mixed(self, race: str):
        """Increment total appearances for mixed race groups."""
        race_node = self._get_race_top_node(race)
        self._add_total_apparences_mixed(race_node)

    def add_race_depth(self, race: str, avg_depth: int):
        """Add average depth for a race."""
        race_node = self._get_race_top_node(race)
        self._add_avg_depth(race_node, avg_depth)

    def add_race_face_position(self, race: str, pos: str):
        """Add face position statistics for a race."""
        race_node = self._get_race_top_node(race)
        position_node = self._get_face_position_node(race_node)
        self._add_first_position(position_node, pos)

    def add_race_in_front_of_count(self, race: str, other_race: str):
        """Increment 'in front of' count for a race vs another race."""
        race_node = self._get_race_top_node(race)
        in_front_node = self._get_in_front_of_count_node(race_node)
        self._add_in_front_of_count(in_front_node, other_race)

    def add_race_in_front_of_val(self, race: str, other_race: str, weighted_val: float):
        """Add weighted 'in front of' value for a race vs another race."""
        race_node = self._get_race_top_node(race)
        in_front_node = self._get_in_front_of_val_node(race_node)
        self._add_in_front_of_val(in_front_node, other_race, weighted_val)

    # =================== gender helper ===================
    def _get_gender_top_node(self, gender: str):
        """Get or initialize the top node for a gender."""
        Top_note = General_Node[Const.num_faces]
        if Const.gender_nodes not in Top_note:
            Top_note[Const.gender_nodes] = {}
        if gender not in Top_note[Const.gender_nodes]:
            Top_note[Const.gender_nodes][gender] = {}
        return Top_note[Const.gender_nodes][gender]

    def add_gender_race(self, gender: str, race: str):
        """Add race statistics under a gender node."""
        gender_node = self._get_gender_top_node(gender)
        race_node = self._get_race_node(gender_node)
        self._add_race(race_dict=race_node, race=race)

    def add_gender_age(self, gender: str, age: int):
        """Add age statistics under a gender node."""
        gender_node = self._get_gender_top_node(gender)
        age_node = self._get_age_node(gender_node)
        self._add_age(age_dict=age_node, age=age)

    def add_gender_emotion(self, gender: str, emotion: str):
        """Add emotion statistics under a gender node."""
        gender_node = self._get_gender_top_node(gender)
        emotion_node = self._get_emotion_node(gender_node)
        self._add_emotion(emotion_dict=emotion_node, emotion=emotion)

    def add_gender_total_num_faces(self, gender: str):
        """Increment total number of faces for a gender."""
        gender_node = self._get_gender_top_node(gender)
        self._add_total_num_faces(gender_node)

    def add_gender_total_num_faces_mixed(self, gender: str):
        """Increment total number of faces for mixed gender groups."""
        gender_node = self._get_gender_top_node(gender)
        self._add_total_num_faces_mixed(gender_node)

    def add_gender_total_apparences(self, gender: str):
        """Increment total appearances for a gender."""
        gender_node = self._get_gender_top_node(gender)
        self._add_total_apparences(gender_node)

    def add_gender_total_apparences_mixed(self, gender: str):
        """Increment total appearances for mixed gender groups."""
        gender_node = self._get_gender_top_node(gender)
        self._add_total_apparences_mixed(gender_node)

    def add_gender_depth(self, gender: str, avg_depth: int):
        """Add average depth for a gender."""
        gender_node = self._get_gender_top_node(gender)
        self._add_avg_depth(gender_node, avg_depth)

    def add_gender_face_position(self, gender: str, pos: str):
        """Add face position statistics for a gender."""
        gender_node = self._get_gender_top_node(gender)
        position_node = self._get_face_position_node(gender_node)
        self._add_first_position(position_node, pos)

    def add_gender_in_front_of_count(self, gender: str, other_gender: str):
        """Increment 'in front of' count for a gender vs another gender."""
        gender_node = self._get_gender_top_node(gender)
        in_front_node = self._get_in_front_of_count_node(gender_node)
        self._add_in_front_of_count(in_front_node, other_gender)

    def add_gender_in_front_of_val(self, gender: str, other_gender: str, weighted_val: float):
        """Add weighted 'in front of' value for a gender vs another gender."""
        gender_node = self._get_gender_top_node(gender)
        in_front_node = self._get_in_front_of_val_node(gender_node)
        self._add_in_front_of_val(in_front_node, other_gender, weighted_val)

    # =================== json helper ===================
    def get_jsons(self) -> str:
        """Return the global statistics as a JSON string."""
        return json.dumps(General_Node)

    def to_json(self, path: str):
        """Write the global statistics to a JSON file."""
        with open(path, "w+") as f:
            json.dump(General_Node, f, sort_keys=True)
