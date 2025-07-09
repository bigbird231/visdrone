
# the meaning of annotation:
# frame_index, target_id, x, y, w, h, score, object_category, truncation, occlusion

# frame_index,
# target_id: used to tracking
# x, y, w, h,
# confidence score: for GT(ground truth), always 1
# object_category: same as sub_class in task1
# truncation: 0 = not truncated, 1 = truncated (part of object outside image)
# occlusion: 0 = fully visible, 1 = partially occluded, 2 = heavily occluded