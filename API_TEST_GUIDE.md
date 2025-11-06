# 高德地图路径规划API直接测试指南

## 方法1：浏览器直接测试（最简单）

### 1. 驾车路径规划
在浏览器地址栏输入以下URL（替换 `YOUR_API_KEY` 为你的高德API密钥）：

```
https://restapi.amap.com/v3/direction/driving?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&extensions=all&output=json
```

**参数说明：**
- `key`: 你的高德API密钥
- `origin`: 起点坐标（经度,纬度），天安门：116.397428,39.90923
- `destination`: 终点坐标（经度,纬度），故宫：116.397026,39.918058
- `extensions`: all（返回详细信息）
- `output`: json（返回JSON格式）

### 2. 步行路径规划
```
https://restapi.amap.com/v3/direction/walking?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&extensions=all&output=json
```

### 3. 骑行路径规划
```
https://restapi.amap.com/v3/direction/bicycling?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&extensions=all&output=json
```

### 4. 公交路径规划
```
https://restapi.amap.com/v3/direction/transit/integrated?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&city=北京&extensions=all&output=json
```

## 方法2：使用curl命令（命令行）

### 驾车路径规划
```bash
curl "https://restapi.amap.com/v3/direction/driving?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&extensions=all&output=json"
```

### 步行路径规划
```bash
curl "https://restapi.amap.com/v3/direction/walking?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&extensions=all&output=json"
```

### 骑行路径规划
```bash
curl "https://restapi.amap.com/v3/direction/bicycling?key=YOUR_API_KEY&origin=116.397428,39.90923&destination=116.397026,39.918058&extensions=all&output=json"
```

## 方法3：使用Postman或类似工具

1. 创建新的GET请求
2. URL: `https://restapi.amap.com/v3/direction/walking`（根据出行方式选择）
3. 添加Query参数：
   - `key`: 你的API密钥
   - `origin`: 116.397428,39.90923
   - `destination`: 116.397026,39.918058
   - `extensions`: all
   - `output`: json
   - `city`: 北京（可选）

## 方法4：使用在线API测试工具

访问：https://lbs.amap.com/api/webservice/guide/api/directioninfo

在高德官方文档页面可以直接测试API。

## 常见坐标参考

- **天安门**: 116.397428,39.90923
- **故宫**: 116.397026,39.918058
- **中关村**: 116.313393,39.984211
- **西直门**: 116.346661,39.942856

## 返回结果说明

成功时返回的JSON包含：
- `status`: "1" 表示成功
- `route.paths[0]`: 路径信息
  - `distance`: 距离（米）
  - `duration`: 时间（秒）
  - `steps`: 路径步骤列表
  - `polyline`: 路径坐标串（可能为空，需要从steps中提取）

## 验证路径正确性

1. **检查距离是否合理**：
   - 天安门到故宫步行：约1-2公里
   - 天安门到故宫驾车：约4-5公里

2. **检查时间是否合理**：
   - 步行速度：约5公里/小时
   - 驾车速度：约30-40公里/小时（城市道路）

3. **检查polyline坐标**：
   - 路径起点应该接近查询起点
   - 路径终点应该接近查询终点

## 常见问题

1. **API返回错误**：
   - 检查API密钥是否正确
   - 检查坐标格式是否正确（经度,纬度）
   - 检查是否超出API调用限制

2. **polyline为空**：
   - 这是正常的，需要从`steps`数组中提取每个步骤的`polyline`并合并

3. **路径不合理**：
   - 检查起点和终点坐标是否正确
   - 尝试使用更精确的坐标（通过地理编码API获取）

