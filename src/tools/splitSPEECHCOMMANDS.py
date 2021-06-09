from pathlib import Path
from shutil import copyfile


dic = {}
val_path = 'dataset/SpeechCommands/speech_commands_v0.02/validation_list.txt'
with open(val_path) as f:
    for line in f.readlines():
        dic[line.rstrip("\n")] = 'val'

test_path = 'dataset/SpeechCommands/speech_commands_v0.02/testing_list.txt'
with open(test_path) as f:
    for line in f.readlines():
        dic[line.rstrip("\n")] = 'test'

root = Path('dataset/SpeechCommands/speech_commands_v0.02')
train_path = root.with_name(root.name + '_train')
train_path.mkdir(parents=True, exist_ok=True)
val_path = root.with_name(root.name + '_val')
val_path.mkdir(parents=True, exist_ok=True)
test_path = root.with_name(root.name + '_test')
test_path.mkdir(parents=True, exist_ok=True)
class_name_list = []
for child in root.iterdir():
    if child.is_dir():
        class_name = child.name
        if class_name[0] != '_':
            class_path = train_path/class_name
            class_path.mkdir(parents=True, exist_ok=True)
            class_path = val_path/class_name
            class_path.mkdir(parents=True, exist_ok=True)
            class_path = test_path/class_name
            class_path.mkdir(parents=True, exist_ok=True)
            class_name_list.append(class_name)

            for file in child.iterdir():
                file_path = file.parent.name + '/' +file.name
                # print(file_path)
                if file_path in dic:
                    if dic[file_path] == 'val':
                        copyfile(file, val_path/file_path)
                        print('val')
                    elif dic[file_path] == 'test':
                        copyfile(file, test_path/file_path)
                        print('test')
                else:
                    copyfile(file, train_path/file_path)
                    # print('train')
