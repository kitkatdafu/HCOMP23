import json


def url_to_image_id_map_helper():
    with open("dataset/Dog3.json") as dog3_handle:
        dog3_data = json.load(dog3_handle)
    url_to_image_id_map = {}
    for idx, image in enumerate(dog3_data["content"]):
        url_to_image_id_map[image["image_url"]] = idx

    for i in range(125):
        url_to_image_id_map["/Birds/{}.jpg".format(i)] = i
        url_to_image_id_map["./Birds/{}.jpg".format(i)] = i
        url_to_image_id_map["../Birds/{}.jpg".format(i)] = i

    return url_to_image_id_map
