
## Object Detection
### Outside NN
`bbox_xy` - bouding box written in the XY format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y].

`bbox_wh` - bouding box written in the WH format: [center_x, center_y, width, height].

`bboxes_xy` - array of bouding boxes written in the XY format.

`bboxes_wh` - array of bouding boxes written in the WH format.

`offset` - difference between ground truth and default bouding box.

`offsets` - array of the such differences.

### Inside NN

`dbox_xy` - default bounding box in the XY format.

`dbox_wh` - default bounding box in the WH format.

`dboxes_xy` - array of default bounding boxes in the XY format.

`dboxes_wh` - array of default bounding boxes in the WH format.