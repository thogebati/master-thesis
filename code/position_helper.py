from const import Const

def rank_persons(values: list[dict], tie_threshold: float = 0.05) -> list[dict]:
    current_rank = 1
    values[0][Const.combined][Const.face_position_per] = current_rank
    values[0][Const.combined][Const.face_position_norm] = current_rank
    
    for i in range(1, len(values)):
        prev_person = values[i-1]
        current_person = values[i]

        # Check if the difference in depth is within the tie threshold
        # This handles cases where people are standing next to each other
        depth_diff = abs(prev_person[Const.combined][Const.mean_depth] - current_person[Const.combined][Const.mean_depth]) 
        
        # We compare the difference to a fraction of the previous person's depth
        if depth_diff <= (prev_person[Const.combined][Const.mean_depth] * tie_threshold):
            # It's a tie, assign the same rank
            current_person[Const.combined][Const.face_position_per] = prev_person[Const.combined][Const.face_position_per]
        else:
            # Not a tie, assign a new rank.
            # The new rank is the current position in the list (i) + 1
            current_rank = i + 1
            current_person[Const.combined][Const.face_position_per] = current_rank
        current_person[Const.combined][Const.face_position_norm] = i + 1
    
    return values