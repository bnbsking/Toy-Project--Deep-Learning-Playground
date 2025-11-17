$projectPath = "C:\Users\James\Desktop\code\Toy-Project--Deep-Learning-Playground"

docker run -it --rm `
  -v "${projectPath}:/app" `
  -w /app `
  --name lapp `
  pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime `
  /bin/bash
