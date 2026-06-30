# EasyMedicalImages
The easy way to manipulate medical images

This is a library that focuses on taking CT scans and transforming them into arrays in python; giving the user easy ways to manipulate them as well. This can easily be used for downstream usage of CT images within a Python script.

### Quick Start:
```python
import EasyMedicalImages as EMI
path_to_folder_of_ct_scans=input("Input path to ct scans: ")
instance=EMI.HighLevel(path_to_folder=repr(path_to_folder_of_ct_scans))
megadatalist=instance.run_on_directory()
print(megadatalist) # This outputs all of the images in that directory as arrays!
```

## Tour of the Library
### run_on_directory()


PyPI link: [PyPI](https://pypi.org/project/easy-medical-images/)