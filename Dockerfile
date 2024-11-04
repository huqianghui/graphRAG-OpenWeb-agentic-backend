# 使用官方 Python 3.10 作为基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 复制到容器中
COPY requirements.txt .

# 安装依赖包
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录下的所有文件复制到容器中
COPY . .

# 暴露端口 8000
EXPOSE 8000

# 运行应用程序
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "8000"]