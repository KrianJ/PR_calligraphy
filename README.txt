# This is a PR project which created for PR final exam

# Idea & Data source
    from: idea from PR teacher's research field, img data provided by PR teacher

    Idea: try to extract 8 direction character to express each calligraphy image(processed), then get the similarity measure by using improved cosine distance.
        After which we can recognize the specific calligraphy img by repeat the algorithm steps mention below.

    Data: Raw images stored in characters directory; Processed images stored in after_process directory.

# Directory & Files:
    > after_process: Processed image vectors.
        > cv_label(mtx): data_label & data_set for cross validation.
        > label_character.csv: each image recorded by (characterID, label, file_name, img) type
        > label_count.csv: record the number of different word.
        > test_data: data for testing
        > unlabel_charater.csv: unlabeled images
    > characters: raw images.
    > shape_match_result: Storing algorithm result.
    > demo.py: A smaller algorithm model which use 20 words' images due to my poor computer.
    > fea_extract.py / main.py / pre_process.py / similarity.py:
        u can get the meaning of those py files when reading their comments.

# TIPS
    Because of the large file upload restrict, the sub-files under directory after_process, cv_mtx.mat and label_character.csv, are not upload to this repository. Besides, 
the directory of raw images called charaters isn't here for the same reason. If needed, i'll use Git LFS to commit them.
