wget "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1751013_student_hcmus_edu_vn/EUqd8S65Y7NBtKA35HUSUl8BBvbZiAG9MUEYj9qO3PMFqw?e=3rb2QW&download=1" -O data/classTraining.cla
wget "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1751013_student_hcmus_edu_vn/EXZ6NIZVGxVBg8QqZZoOVv0BetN1ynopLLDKtVJzAgJ_5Q?e=NahuaA&download=1" -O data/OFF_training_new.tgz
wget "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1751013_student_hcmus_edu_vn/EVu9T4u_i0RIhbtgIoEYiPoBmAjHwo9MDElFUuty77XrOA?e=rErFCA&download=1" -O data/OFF_test_new.tgz
wget "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1751013_student_hcmus_edu_vn/EdjCBovFKR1Ot17b35sIHOABMKtCS7BGf8FMMAJzLohiMQ?e=raDZlo&download=1" -O data/PROP_test_new.tgz
wget "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1751013_student_hcmus_edu_vn/ETLF8A1TG1BFliTcoJ4_OZwB8tTc3htG6kVtpu5LDN7Xsw?e=sBX6hz&download=1" -O data/PROP_training_new.tgz
cd /content/data
tar -xvzf PROP_training_new.tgz
tar -xvzf PROP_test_new.tgz
tar -xvzf OFF_training_new.tgz
tar -xvzf OFF_test_new.tgz
cd /content