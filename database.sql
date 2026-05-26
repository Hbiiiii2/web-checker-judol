-- 1. Membuat Database Baru (Jika belum ada)
CREATE DATABASE IF NOT EXISTS web_checker;
USE web_checker;

-- 2. Membuat Tabel Users
-- Menyimpan informasi akun pengguna dan status langganan premium
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    is_premium TINYINT(1) DEFAULT 0, -- 0 = Free, 1 = Premium
    package_type VARCHAR(20) DEFAULT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Membuat Tabel Payments
-- Menyimpan histori transaksi dari Midtrans yang berhasil dibayar
CREATE TABLE IF NOT EXISTS payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    transaction_id VARCHAR(255) NOT NULL UNIQUE,
    package_type VARCHAR(20),
    payment_type VARCHAR(50),
    gross_amount DECIMAL(10, 2),
    transaction_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;