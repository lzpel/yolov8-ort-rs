use image::{DynamicImage, GenericImageView, Pixel};
use ort::{SessionBuilder, Value};

pub fn yolo(model_path: impl AsRef<std::path::Path>, original_img: &image::DynamicImage, confidence: f32) -> ort::Result<(DynamicImage, Vec<BBox>)> {
    let size = 640usize;
    let model = SessionBuilder::new()?.commit_from_file(model_path)?;
    let img = resize_with_padding(size, original_img);
    //https://github.com/pykeio/ort/blob/main/examples/yolov8/examples/yolov8.rs
    // show model info
    println!("{:?}", model.inputs);
    println!("{:?}", model.outputs);
    let input: Vec<f32> = [0, 1, 2].into_iter().map(|v| img.pixels().map(move |(_x, _y, c)| c.channels()[v] as f32 / 255.)).flatten().collect();
    let input_tensor = Value::from_array(([1, 3, size as usize, size as usize], input))?;
    let outputs = model.run(ort::inputs!["images" => input_tensor]?)?;
    let (_key, raw_output) = outputs.first_key_value().unwrap();
    let output = raw_output.try_extract_tensor::<f32>()?.t().into_owned();
    let output_shape = output.shape();
    println!("{:?}", &output_shape);
    let output_reshaped = output.to_shape((output_shape[0], output_shape[1])).expect("Failed to reshape the output");
    let mut boxes: Vec<BBox> = Default::default();
    for row in output_reshaped.axis_iter(output_reshaped.axes().nth(0).unwrap().axis) {
        let row: Vec<_> = row.iter().copied().collect();
        let (class, prob) = row
            .iter()
            .skip(4) // skip bounding box coordinates
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < confidence {
            continue;
        }
        boxes.push(BBox::new([0, 1, 2, 3].map(|v| row[v] / size as f32), prob, class));
    }
    Ok((img, BBox::nms(boxes, 0.45)))
}
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub probability: f32,
    pub class: usize,
}
impl BBox {
    pub fn new([xc, yc, w, h]: [f32; 4], probability: f32, class: usize) -> Self {
        Self {
            x1: xc - w / 2.,
            y1: yc - h / 2.,
            x2: xc + w / 2.,
            y2: yc + h / 2.,
            probability,
            class,
        }
    }
    pub fn intersection(&self, other: &Self) -> f32 {
        (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0) * (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0)
    }
    pub fn union(&self, other: &Self) -> f32 {
        ((self.x2 - self.x1) * (self.y2 - self.y1)) + ((other.x2 - other.x1) * (other.y2 - other.y1)) - self.intersection(other)
    }
    pub fn iou(&self, other: &Self) -> f32 {
        self.intersection(other) / self.union(other)
    }
    pub fn quantized_box(&self, width: usize, height: usize) -> [usize; 4] {
        [(self.x1, width), (self.y1, height), (self.x2, width), (self.y2, height)].map(|v| ((v.0 * v.1 as f32) as usize).min(v.1 - 1).max(0))
    }
    pub fn nms(boxes: Vec<Self>, iou_threshold: f32) -> Vec<Self> {
        let mut sorted_boxes = boxes.clone();
        sorted_boxes.sort_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap());
        let mut result = Vec::new();
        while let Some(best_box) = sorted_boxes.pop() {
            result.push(best_box);
            sorted_boxes.retain(|bbox| best_box.iou(&bbox) < iou_threshold);
        }
        result
    }
}
pub fn resize_with_padding(size: usize, src: &image::DynamicImage) -> image::DynamicImage {
    let mut dst = image::ImageBuffer::from_pixel(size as u32, size as u32, image::Rgb([255, 255, 255]));
    let resized = src.resize(size as u32, size as u32, image::imageops::CatmullRom);
    let (x_offset, y_offset) = ((dst.width() - resized.width()) / 2, (dst.height() - resized.height()) / 2);
    resized.pixels().for_each(|(x, y, pixel)| dst.put_pixel(x + x_offset, y + y_offset, pixel.to_rgb()));
    image::DynamicImage::from(dst)
}
pub fn draw_bbox(src: &image::DynamicImage, bbox: &Vec<BBox>) -> image::DynamicImage {
    let mut dst = src.clone().into_rgb8();
    for i in bbox {
        let [x1, y1, x2, y2] = i.quantized_box(src.width() as usize, src.height() as usize);
        let color = image::Rgb([0, 1, 2].map(|v| ((i.class + 1) >> v) & 1 != 0).map(|v| if v { 255 } else { 50 }));
        for x in x1..x2 {
            dst.put_pixel(x as u32, y1 as u32, color);
            dst.put_pixel(x as u32, y2 as u32, color);
        }
        for y in y1..y2 {
            dst.put_pixel(x1 as u32, y as u32, color);
            dst.put_pixel(x2 as u32, y as u32, color);
        }
    }
    dst.into()
}

fn main() {
    let img_source= image::open(r"bus.jpg").unwrap();
    let (img, out) = yolo("yolov8n.onnx", &img_source, 0.5).expect("TODO: panic message");
    draw_bbox(&img, &out).save("./output.png").expect("TODO: panic message");
}