# sears
Code for [Semantically Equivalent Adversarial Rules for Debugging NLP Models](https://homes.cs.washington.edu/~marcotcr/acl18.pdf)

# Installation
Run the following:
```
git clone https://github.com/SebOchs/sears.git
cd sears
virtualenv -p python3 ENV
source ENV/bin/activate
pip install editdistance keras numpy jupyter tensorflow-gpu==1.3.0 torchtext==0.1.1 spacy==1.9.0
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
python -m ipykernel install --user --name=sears
python -m spacy download en
git clone https://github.com/SebOchs/OpenNMT-py
cd OpenNMT-py/
python setup.py install
cd ..
```

In cooperation with Anna Filighera and the Multimedia Communications Lab at TU Darmstadt.
