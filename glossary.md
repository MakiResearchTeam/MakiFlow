
## Object Detection
### Outside NN
`bbox_xy` - bounding box written in the XY format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y].

`bbox_wh` - bounding box written in the WH format: [center_x, center_y, width, height].

`bboxes_xy` - array of bounding boxes written in the XY format.

`bboxes_wh` - array of bounding boxes written in the WH format.

`gbox_xy` - ground truth bounding box written in the XY format.

`gbox_wh` - ground truth bounding box written in the WH format.

`gboxes_xy` - array of ground truth bounding boxes written in the XY format.

`gboxes_wh` - array of ground truth bounding boxes written in the WH format.

`offset` - vector of difference between ground truth and default bounding box.

`offsets` - array of the such differences.

### Inside NN

`dbox_xy` - default bounding box in the XY format.

`dbox_wh` - default bounding box in the WH format.

`dboxes_xy` - array of default bounding boxes in the XY format.

`dboxes_wh` - array of default bounding boxes in the WH format.