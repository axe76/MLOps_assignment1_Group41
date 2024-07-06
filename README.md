# Task 2 DVC Versioning

## Structure

Folder ```local``` represents local developer's machine. It contains the ```.csv.dvc``` file with the correct hash code to fetch the latest version of the dataset.<br>

Folder ```remote``` represents the remote storage where the dataset is hosted and has been included in the repo for readability.

## Data Versions
There are 2 data versions available in the ```remote``` folder, each with its own hash. The ```.csv.dvc``` in the ```local``` folder can be updated with the required hash after which the ```dvc pull``` command will fetch the respective dataset version.<br>

1st version hash: ```03ee85e53ee8d272c7ee492f714e3b1b``` <br>
2nd version hash: ```29aa6c5003152d07f31b3f2e7bf7af24``` <br>
