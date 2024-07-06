# Task 2 DVC Versioning

## Structure

Folder ```local``` represents local developer's machine. It contains the ```.csv.dvc``` file with the correct hash code to fetch the latest version of the dataset.<br>

Folder ```remote``` represents the remote storage where the dataset is hosted and has been included in the repo for readability.

## Data Versions
There are 2 data versions available in the ```remote``` folder, each with its own hash. The ```.csv.dvc``` can be updated with the required hash after which the ```dvc pull``` command will fetch the respective dataset version.
