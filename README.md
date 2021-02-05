
# AWS Lambda + Docker 

The code for the workshop I made on deploying Keras models with AWS Lambda and Docker

* Detailed guide: [guide.md](guide.md)

Want to talk about it? Join [DataTalks.Club](https://datatalks.club)


## Running it

Download the model:

```bash
wget https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5
```

Convert the model to tf-lite:

```bash
python keras_to_tf.py
```

Build the image:

(Important: don't change the workdir in Dockerfile)

```
docker build -t tf-lite-lambda .
```

Run it:

```bash
docker run --rm \
    -p 8080:8080 \
    tf-lite-lambda
```

Test:

```bash
URL="http://localhost:8080/2015-03-31/functions/function/invocations"

REQUEST='{
    "url": "http://bit.ly/mlbookcamp-pants"
}'

curl -X POST \
    -H "Content-Type: application/json" \
    --data "${REQUEST}" \
    "${URL}" | jq
```

Response:


```json
{
  "dress": -1.8682900667190552,
  "hat": -4.7612457275390625,
  "longsleeve": -2.3169822692871094,
  "outwear": -1.062570571899414,
  "pants": 9.88715648651123,
  "shirt": -2.8124303817749023,
  "shoes": -3.66628360748291,
  "shorts": 3.2003610134124756,
  "skirt": -2.6023387908935547,
  "t-shirt": -4.835044860839844
}
```


Create a registry in ECR:

```
aws ecr create-repository --repository-name lambda-images 
```

Push to ECR

```
ACCOUNT=XXXXXXXXXXXX

docker tag tf-lite-lambda ${ACCOUNT}.dkr.ecr.eu-west-1.amazonaws.com/lambda-images:tf-lite-lambda

$(aws ecr get-login --no-include-email)

docker push ${ACCOUNT}.dkr.ecr.eu-west-1.amazonaws.com/lambda-images:tf-lite-lambda
```

Now use the URI of the image to create a Lambda function. Give it enough RAM (1024MB is good).

