# data/ (not committed)

```
data/
  hagrid_yolo/            made by scripts/prepare_hagrid.py
    images/{train,val,test}/<class>_<id>.jpg
    labels/{train,val,test}/<class>_<id>.txt
    meta.csv              split, file, gesture, user_id, width, height, n_boxes
    data.yaml
    split_users.json      subject ids per split (subject-disjoint)
  classroom/              team-collected volunteer test set (never trained on, never published)
    images/s07_like_low_light_1.jpg
    labels.csv            file, subject_id, gesture, condition, background_gesture
  lfw/                    LFW images + pairsDevTrain.txt / pairsDevTest.txt (face threshold only)
  app/                    attendance.db, secret.key, instructor_key.txt (created by the app)
```

`labels.csv` for the classroom set (the `/collect` page writes it for you):

| column | meaning |
|---|---|
| file | path relative to `data/classroom` |
| subject_id | volunteer id (identity) |
| gesture | gesture the volunteer shows (`like, peace, palm, fist, ok, one`) or `none` |
| condition | `normal`, `low_light`, `far_crowded` (any text works; results are grouped by it) |
| background_gesture | optional: gesture shown by **someone else** in the background (background-hand attack images) |

Suggested protocol per volunteer (~18 images + a few attack images): each of the 6 gestures × 3 conditions,
plus 2–3 images where the volunteer shows no gesture / a different gesture while a teammate in the
background shows another one (`background_gesture`).
