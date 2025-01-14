import os
from bing_image_downloader import downloader

def crawl_tomato_images(output_folder="data_lake", num_images=100):
    # Đảm bảo thư mục output tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Từ khóa tìm kiếm
    search_term = "tomato"
    
    print(f"Đang tải {num_images} ảnh từ Bing với từ khóa: {search_term}...")
    
    # Tải ảnh
    downloader.download(
        search_term,
        limit=num_images,
        output_dir=output_folder,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
    
    print(f"Ảnh đã được lưu trong thư mục: {output_folder}/{search_term}")

if __name__ == "__main__":
    # Gọi hàm để tải ảnh
    crawl_tomato_images()
