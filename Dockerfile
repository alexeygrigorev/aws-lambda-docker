FROM public.ecr.aws/lambda/python:3.7

COPY tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl

RUN pip3 install --upgrade pip

RUN pip3 install \
    numpy==1.16.5 \
    Pillow==6.2.1 \ 
    tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl \
    --no-cache-dir

COPY lambda_function.py lambda_function.py

CMD [ "lambda_function.lambda_handler" ]
