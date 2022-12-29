# Тестовое задание в Сloud

Перед запуском обучения требуется установить зависимости из файла `requirements.txt`. Затем запустить `train/main.py`. 

Чекпоинты сохраняются в папку `./W`, логи для просмотра в tensorboard сохраняются в папку `./tb_logs`. 

Для дообучения модели необходимо подать параметр `--checkpoint_load_path "path to weight"`. 

Дообучить на новых данных можно указав путь к новому датасету в анатации COCO `--image_train_path "path to images"` и `--annotation_train_path "path to annotation"`.

Например, чтобы дообучить веса на новых данных с десятью классами и сохранить чекпоинты под новым названием требуется выполнить команду
```
python3 train/main.py --checkpoint_load_path ./W/fasterrcnn.ckpt --num_classes 10 --image_train_path ./images/train2017/ --annotation_train_path ./annotations/instances_train2017.json --model_name fasterrcnn_V2 
```

Для сборки и запуска докера требуется выполнить следующие команды в папке `./docker`
```
docker build -t fasterrcnn .
docker run -p 8000:8000 fasterrcnn
```

Перед запуском докера требуется скомпилировать обученную модель (или скомпилировать модель которую обучили кодом выше)
```
import torchvision
import torch
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("./docker/fasterrcnn.pt")

```

Докер поднимет локальный веб сервер в котором можно получить результат такого вида:
![Пример](https://github.com/tot1342/claud_test_task/blob/main/docker/example_work.png)

