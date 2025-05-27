import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import hashlib

class EarDiagnosisSystem:
    def __init__(self):
        # File paths untuk JSON storage
        self.data_dir = "data"
        self.data_file = os.path.join(self.data_dir, "ear_diagnosis_data.json")
        self.users_file = os.path.join(self.data_dir, "admin_users.json")
        self.stats_file = os.path.join(self.data_dir, "consultation_stats.json")
        
        # Buat folder data jika belum ada
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load data dari JSON atau buat default
        self.load_data()
        self.consultation_count = 0
        self.disease_stats = {}
        self.load_stats()
        
        # Session management
        self.current_user = None
        self.is_admin_logged_in = False
        self.is_user_logged_in = False

    def load_data(self):
        """Load data dari file JSON"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.diseases = data.get('diseases', {})
                    self.symptoms = data.get('symptoms', {})
            except Exception as e:
                print(f"Error loading data: {e}")
                self.create_default_data()
        else:
            self.create_default_data()

    def save_data(self):
        """Simpan data ke file JSON"""
        data = {
            'diseases': self.diseases,
            'symptoms': self.symptoms,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    def load_stats(self):
        """Load statistik konsultasi"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    self.consultation_count = stats.get('consultation_count', 0)
                    self.disease_stats = stats.get('disease_stats', {})
            except Exception as e:
                print(f"Error loading stats: {e}")

    def save_stats(self):
        """Simpan statistik konsultasi"""
        stats = {
            'consultation_count': self.consultation_count,
            'disease_stats': self.disease_stats,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

    def create_default_data(self):
        """Buat data default jika belum ada"""
        self.diseases = {
            'P01': {
                'name': 'Otitis Eksterna (Swimmer\'s Ear)',
                'symptoms': ['G01', 'G02', 'G03', 'G04', 'G05'],
                'info': 'Peradangan pada saluran telinga luar yang sering disebabkan oleh infeksi bakteri atau jamur.',
                'solution': 'Hindari masuknya air ke telinga, gunakan obat tetes telinga sesuai resep dokter, dan jaga kebersihan telinga. Segera konsultasi ke dokter THT.'
            },
            'P02': {
                'name': 'Otitis Media Akut',
                'symptoms': ['G06', 'G07', 'G08', 'G09', 'G10'],
                'info': 'Peradangan telinga tengah yang sering terjadi pada anak-anak, disebabkan oleh infeksi bakteri atau virus.',
                'solution': 'Istirahat yang cukup, kompres hangat, obat pereda nyeri sesuai dosis, dan antibiotik jika diperlukan. Konsultasi dengan dokter THT untuk penanganan lebih lanjut.'
            },
            'P03': {
                'name': 'Serumen Prop (Kotoran Telinga Mengeras)',
                'symptoms': ['G11', 'G12', 'G13', 'G14'],
                'info': 'Penumpukan kotoran telinga yang mengeras dan menyumbat saluran telinga.',
                'solution': 'Jangan membersihkan dengan cotton bud. Gunakan obat pelembut serumen atau konsultasi ke dokter untuk pembersihan profesional.'
            },
            'P04': {
                'name': 'Tinnitus (Telinga Berdenging)',
                'symptoms': ['G15', 'G16', 'G11', 'G17'],
                'info': 'Sensasi mendengar suara berdenging, berdesis, atau berdesir tanpa sumber suara eksternal.',
                'solution': 'Hindari kebisingan berlebihan, kelola stress, cukup tidur, dan konsultasi dengan dokter untuk mengetahui penyebab dasarnya.'
            },
            'P05': {
                'name': 'Vertigo Perifer',
                'symptoms': ['G18', 'G16', 'G10', 'G17'],
                'info': 'Gangguan keseimbangan yang berasal dari telinga bagian dalam, menyebabkan sensasi berputar.',
                'solution': 'Istirahat dalam posisi yang nyaman, hindari gerakan mendadak, minum obat anti vertigo sesuai resep dokter.'
            },
            'P06': {
                'name': 'Barotrauma Telinga',
                'symptoms': ['G06', 'G11', 'G02', 'G13'],
                'info': 'Cedera telinga akibat perubahan tekanan udara yang tiba-tiba, sering terjadi saat naik pesawat.',
                'solution': 'Lakukan manuver Valsalva dengan hati-hati, kunyah permen karet, atau konsultasi dokter jika nyeri berlanjut.'
            }
        }

        self.symptoms = {
            'G01': 'Nyeri telinga yang tajam dan berdenyut',
            'G02': 'Gatal pada telinga',
            'G03': 'Keluarnya cairan dari telinga',
            'G04': 'Pembengkakan di sekitar telinga',
            'G05': 'Sensitivitas saat menyentuh telinga',
            'G06': 'Nyeri telinga yang dalam',
            'G07': 'Demam tinggi',
            'G08': 'Telinga terasa penuh',
            'G09': 'Gangguan pendengaran sementara',
            'G10': 'Mual atau muntah',
            'G11': 'Pendengaran berkurang',
            'G12': 'Telinga terasa tersumbat',
            'G13': 'Sensasi penuh di telinga',
            'G14': 'Bau tidak sedap dari telinga',
            'G15': 'Telinga berdenging terus-menerus',
            'G16': 'Pusing atau vertigo',
            'G17': 'Sulit berkonsentrasi',
            'G18': 'Kehilangan keseimbangan'
        }
        
        self.save_data()

    def load_users(self):
        """Load data users"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading users: {e}")
                return {}
        else:
            # Buat default users
            default_users = {
                'admin': {
                    'password_hash': self.hash_password('admin123'),
                    'role': 'admin',
                    'created_at': datetime.now().isoformat()
                },
                'user': {
                    'password_hash': self.hash_password('user123'),
                    'role': 'user',
                    'created_at': datetime.now().isoformat()
                }
            }
            self.save_users(default_users)
            return default_users

    def save_users(self, users_data):
        """Simpan data users"""
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving users: {e}")
            return False

    def hash_password(self, password):
        """Hash password menggunakan SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_login(self, username, password):
        """Verifikasi login"""
        users = self.load_users()
        if username in users:
            password_hash = self.hash_password(password)
            if users[username]['password_hash'] == password_hash:
                self.current_user = username
                if users[username]['role'] == 'admin':
                    self.is_admin_logged_in = True
                    self.is_user_logged_in = True
                else:
                    self.is_user_logged_in = True
                    self.is_admin_logged_in = False
                return True, "Login berhasil!", users[username]['role']
        return False, "Username atau password salah!", None

    def logout(self):
        """Logout semua user"""
        self.current_user = None
        self.is_admin_logged_in = False
        self.is_user_logged_in = False
        return "Logout berhasil!"

    # CRUD Operations untuk Gejala
    def add_symptom(self, symptom_id, symptom_desc):
        """Tambah gejala baru"""
        if not self.is_admin_logged_in:
            return "❌ Akses ditolak! Silakan login sebagai admin terlebih dahulu."
        
        if not symptom_id or not symptom_desc:
            return "❌ ID Gejala dan Deskripsi tidak boleh kosong!"
        
        if symptom_id in self.symptoms:
            return f"❌ Gejala dengan ID {symptom_id} sudah ada!"
        
        self.symptoms[symptom_id] = symptom_desc
        if self.save_data():
            return f"✅ Gejala {symptom_id} berhasil ditambahkan!"
        else:
            return "❌ Gagal menyimpan data!"

    def update_symptom(self, symptom_id, new_desc):
        """Update gejala"""
        if not self.is_admin_logged_in:
            return "❌ Akses ditolak! Silakan login sebagai admin terlebih dahulu."
        
        if symptom_id not in self.symptoms:
            return f"❌ Gejala dengan ID {symptom_id} tidak ditemukan!"
        
        if not new_desc:
            return "❌ Deskripsi tidak boleh kosong!"
        
        old_desc = self.symptoms[symptom_id]
        self.symptoms[symptom_id] = new_desc
        if self.save_data():
            return f"✅ Gejala {symptom_id} berhasil diupdate!\nLama: {old_desc}\nBaru: {new_desc}"
        else:
            return "❌ Gagal menyimpan data!"

    def delete_symptom(self, symptom_id):
        """Hapus gejala"""
        if not self.is_admin_logged_in:
            return "❌ Akses ditolak! Silakan login sebagai admin terlebih dahulu."
        
        if symptom_id not in self.symptoms:
            return f"❌ Gejala dengan ID {symptom_id} tidak ditemukan!"
        
        # Cek apakah gejala digunakan di penyakit
        used_in_diseases = []
        for disease_id, disease in self.diseases.items():
            if symptom_id in disease['symptoms']:
                used_in_diseases.append(disease['name'])
        
        if used_in_diseases:
            return f"❌ Gejala tidak dapat dihapus karena masih digunakan di penyakit: {', '.join(used_in_diseases)}"
        
        deleted_symptom = self.symptoms.pop(symptom_id)
        if self.save_data():
            return f"✅ Gejala {symptom_id} ({deleted_symptom}) berhasil dihapus!"
        else:
            return "❌ Gagal menyimpan data!"

    # CRUD Operations untuk Penyakit
    def add_disease(self, disease_id, disease_name, disease_info, disease_solution, symptoms_list):
        """Tambah penyakit baru"""
        if not self.is_admin_logged_in:
            return "❌ Akses ditolak! Silakan login sebagai admin terlebih dahulu."
        
        if not all([disease_id, disease_name, disease_info, disease_solution]):
            return "❌ Semua field harus diisi!"
        
        if disease_id in self.diseases:
            return f"❌ Penyakit dengan ID {disease_id} sudah ada!"
        
        # Parse symptoms list
        if symptoms_list:
            symptom_ids = [s.strip() for s in symptoms_list.split(',')]
            # Validasi symptom IDs
            invalid_symptoms = [s for s in symptom_ids if s not in self.symptoms]
            if invalid_symptoms:
                return f"❌ Gejala tidak valid: {', '.join(invalid_symptoms)}"
        else:
            symptom_ids = []
        
        self.diseases[disease_id] = {
            'name': disease_name,
            'symptoms': symptom_ids,
            'info': disease_info,
            'solution': disease_solution
        }
        
        if self.save_data():
            return f"✅ Penyakit {disease_name} berhasil ditambahkan!"
        else:
            return "❌ Gagal menyimpan data!"

    def update_disease(self, disease_id, disease_name, disease_info, disease_solution, symptoms_list):
        """Update penyakit"""
        if not self.is_admin_logged_in:
            return "❌ Akses ditolak! Silakan login sebagai admin terlebih dahulu."
        
        if disease_id not in self.diseases:
            return f"❌ Penyakit dengan ID {disease_id} tidak ditemukan!"
        
        # Parse symptoms list
        if symptoms_list:
            symptom_ids = [s.strip() for s in symptoms_list.split(',')]
            # Validasi symptom IDs
            invalid_symptoms = [s for s in symptom_ids if s not in self.symptoms]
            if invalid_symptoms:
                return f"❌ Gejala tidak valid: {', '.join(invalid_symptoms)}"
        else:
            symptom_ids = []
        
        self.diseases[disease_id].update({
            'name': disease_name or self.diseases[disease_id]['name'],
            'info': disease_info or self.diseases[disease_id]['info'],
            'solution': disease_solution or self.diseases[disease_id]['solution'],
            'symptoms': symptom_ids if symptoms_list else self.diseases[disease_id]['symptoms']
        })
        
        if self.save_data():
            return f"✅ Penyakit {disease_id} berhasil diupdate!"
        else:
            return "❌ Gagal menyimpan data!"

    def delete_disease(self, disease_id):
        """Hapus penyakit"""
        if not self.is_admin_logged_in:
            return "❌ Akses ditolak! Silakan login sebagai admin terlebih dahulu."
        
        if disease_id not in self.diseases:
            return f"❌ Penyakit dengan ID {disease_id} tidak ditemukan!"
        
        deleted_disease = self.diseases.pop(disease_id)
        if self.save_data():
            return f"✅ Penyakit {deleted_disease['name']} berhasil dihapus!"
        else:
            return "❌ Gagal menyimpan data!"

    # Fungsi untuk menampilkan data
    def get_symptoms_list(self):
        """Ambil daftar gejala"""
        if not self.symptoms:
            return "Tidak ada gejala yang tersedia."
        
        result = "📋 **Daftar Gejala:**\n\n"
        for code, desc in self.symptoms.items():
            result += f"• **{code}**: {desc}\n"
        return result

    def get_diseases_list(self):
        """Ambil daftar penyakit"""
        if not self.diseases:
            return "Tidak ada penyakit yang tersedia."
        
        result = "🏥 **Daftar Penyakit:**\n\n"
        for code, disease in self.diseases.items():
            result += f"### {code}: {disease['name']}\n"
            result += f"**Info**: {disease['info']}\n"
            result += f"**Gejala**: {', '.join(disease['symptoms'])}\n"
            result += f"**Solusi**: {disease['solution']}\n\n"
        return result

    def get_consultation_stats(self):
        """Ambil statistik konsultasi"""
        result = f"📊 **Statistik Konsultasi:**\n\n"
        result += f"• Total konsultasi: {self.consultation_count}\n"
        result += f"• Total penyakit: {len(self.diseases)}\n"
        result += f"• Total gejala: {len(self.symptoms)}\n\n"
        
        if self.disease_stats:
            result += "**Diagnosa Terpopuler:**\n"
            sorted_stats = sorted(self.disease_stats.items(), key=lambda x: x[1], reverse=True)
            for disease, count in sorted_stats[:5]:
                result += f"• {disease}: {count} kali\n"
        
        return result

    def process_diagnosis(self, *args):
        if not self.is_user_logged_in:
            return "❌ Silakan login terlebih dahulu!", "", "", ""
            
        selected_symptoms = []
        
        for i, is_selected in enumerate(args):
            symptom_code = f'G{str(i+1).zfill(2)}'
            if symptom_code in self.symptoms and is_selected:
                selected_symptoms.append(symptom_code)

        if not selected_symptoms:
            return "❌ Silakan pilih minimal satu gejala terlebih dahulu!", "", "", ""

        results = []
        
        for disease_code, disease in self.diseases.items():
            matched_symptoms = 0
            matching_symptom_list = []
            
            for symptom_code in disease['symptoms']:
                if symptom_code in selected_symptoms:
                    matched_symptoms += 1
                    matching_symptom_list.append(symptom_code)
            
            if matched_symptoms > 0:
                confidence = (matched_symptoms / len(disease['symptoms'])) * 100
                
                results.append({
                    'code': disease_code,
                    'name': disease['name'],
                    'info': disease['info'],
                    'solution': disease['solution'],
                    'matching_symptoms': matching_symptom_list,
                    'confidence': round(confidence, 1),
                    'total_symptoms': len(disease['symptoms'])
                })

        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.consultation_count += 1
        if results:
            top_disease = results[0]['name']
            self.disease_stats[top_disease] = self.disease_stats.get(top_disease, 0) + 1
        
        self.save_stats()
        return self.format_results(selected_symptoms, results)

    def format_results(self, selected_symptoms, results):
        selected_text = "📋 **Gejala yang Anda Pilih:**\n\n"
        for code in selected_symptoms:
            symptom_name = self.symptoms.get(code, "Unknown")
            selected_text += f"• {code}: {symptom_name}\n"

        if not results:
            diagnosis_text = "🤔 **Tidak ditemukan penyakit yang sesuai dengan gejala yang dipilih.**\n\nSilakan konsultasikan dengan dokter untuk diagnosis lebih akurat."
            solution_text = ""
            stats_text = f"📊 **Statistik Konsultasi:** {self.consultation_count} total konsultasi"
        else:
            diagnosis_text = "🎯 **Hasil Diagnosis:**\n\n"
            
            for i, result in enumerate(results):
                rank_emoji = "🏆" if i == 0 else f"{i+1}."
                diagnosis_text += f"{rank_emoji} **{result['name']}** - {result['confidence']}%\n"
                diagnosis_text += f"   📝 {result['info']}\n"
                diagnosis_text += f"   🎯 Gejala cocok: {len(result['matching_symptoms'])}/{result['total_symptoms']}\n\n"

            primary_result = results[0]
            solution_text = f"💡 **Rekomendasi untuk {primary_result['name']}:**\n\n{primary_result['solution']}\n\n"
            solution_text += "⚠️ **Disclaimer:** Sistem ini hanya sebagai alat bantu diagnosis awal. Untuk penanganan yang tepat, selalu konsultasikan dengan dokter spesialis THT."

            stats_text = f"📊 **Statistik:**\n"
            stats_text += f"• Total konsultasi: {self.consultation_count}\n"
            stats_text += f"• Gejala yang dipilih: {len(selected_symptoms)}\n"
            stats_text += f"• Tingkat keyakinan tertinggi: {results[0]['confidence']}%"

        return selected_text, diagnosis_text, solution_text, stats_text

# Inisialisasi sistem
system = EarDiagnosisSystem()

def create_gradio_interface():
    with gr.Blocks(title="Sistem Pakar Diagnosa Penyakit Telinga", theme=gr.themes.Soft()) as demo:
        # Define all UI components first
        with gr.Column(visible=True) as login_screen:
            gr.Markdown("# 🔐 Login Sistem Diagnosa Telinga")
            gr.Markdown("*Silakan login menggunakan akun Anda*")
            
            with gr.Row():
                username_input = gr.Textbox(label="Username", placeholder="Masukkan username")
                password_input = gr.Textbox(label="Password", type="password", placeholder="Masukkan password")
            
            login_btn = gr.Button("🔐 Login", variant="primary")
            login_status = gr.Markdown("Silakan masukkan username dan password Anda")

        # Admin Dashboard Components
        with gr.Column(visible=False) as admin_dashboard:
            gr.Markdown("# 🩺 Sistem Pakar Diagnosa Penyakit Telinga (Admin)")
            current_user_display = gr.Markdown()
            
            logout_btn = gr.Button("🚪 Logout", variant="secondary")
            
            with gr.Tabs():
                with gr.TabItem("🔍 Konsultasi Diagnosis"):
                    gr.Markdown("## Pilih gejala yang Anda rasakan:")
                    
                    inputs = []
                    
                    with gr.Column():
                        for code, symptom_name in system.symptoms.items():
                            symptom_check = gr.Checkbox(
                                label=f"{code}: {symptom_name}",
                                value=False
                            )
                            inputs.append(symptom_check)
                    
                    process_btn = gr.Button("🧮 Proses Diagnosis", variant="primary", size="lg")
                    
                    with gr.Column():
                        selected_output = gr.Markdown(label="Gejala Terpilih")
                        diagnosis_output = gr.Markdown(label="Hasil Diagnosis")
                        solution_output = gr.Markdown(label="Rekomendasi")
                        stats_output = gr.Markdown(label="Statistik")
                    
                    process_btn.click(
                        fn=system.process_diagnosis,
                        inputs=inputs,
                        outputs=[selected_output, diagnosis_output, solution_output, stats_output]
                    )

                with gr.TabItem("📝 Kelola Gejala"):
                    gr.Markdown("## Manajemen Gejala")
                    
                    with gr.Tabs():
                        with gr.TabItem("➕ Tambah Gejala"):
                            add_symptom_id = gr.Textbox(label="ID Gejala", placeholder="G19")
                            add_symptom_desc = gr.Textbox(label="Deskripsi Gejala", placeholder="Deskripsi gejala...")
                            add_symptom_btn = gr.Button("➕ Tambah Gejala", variant="primary")
                            add_symptom_result = gr.Markdown()
                            
                            add_symptom_btn.click(
                                fn=system.add_symptom,
                                inputs=[add_symptom_id, add_symptom_desc],
                                outputs=add_symptom_result
                            )
                        
                        with gr.TabItem("✏️ Edit Gejala"):
                            edit_symptom_id = gr.Textbox(label="ID Gejala", placeholder="G01")
                            edit_symptom_desc = gr.Textbox(label="Deskripsi Baru", placeholder="Deskripsi gejala baru...")
                            edit_symptom_btn = gr.Button("✏️ Update Gejala", variant="secondary")
                            edit_symptom_result = gr.Markdown()
                            
                            edit_symptom_btn.click(
                                fn=system.update_symptom,
                                inputs=[edit_symptom_id, edit_symptom_desc],
                                outputs=edit_symptom_result
                            )
                        
                        with gr.TabItem("🗑️ Hapus Gejala"):
                            delete_symptom_id = gr.Textbox(label="ID Gejala", placeholder="G01")
                            delete_symptom_btn = gr.Button("🗑️ Hapus Gejala", variant="stop")
                            delete_symptom_result = gr.Markdown()
                            
                            delete_symptom_btn.click(
                                fn=system.delete_symptom,
                                inputs=[delete_symptom_id],
                                outputs=delete_symptom_result
                            )
                        
                        with gr.TabItem("📋 Lihat Semua Gejala"):
                            view_symptoms_btn = gr.Button("🔄 Refresh Daftar Gejala")
                            symptoms_list = gr.Markdown()
                            
                            view_symptoms_btn.click(
                                fn=system.get_symptoms_list,
                                outputs=symptoms_list
                            )

                with gr.TabItem("🏥 Kelola Penyakit"):
                    gr.Markdown("## Manajemen Penyakit")
                    
                    with gr.Tabs():
                        with gr.TabItem("➕ Tambah Penyakit"):
                            add_disease_id = gr.Textbox(label="ID Penyakit", placeholder="P07")
                            add_disease_name = gr.Textbox(label="Nama Penyakit", placeholder="Nama penyakit...")
                            add_disease_info = gr.Textbox(label="Informasi Penyakit", lines=3, placeholder="Deskripsi penyakit...")
                            add_disease_solution = gr.Textbox(label="Solusi/Pengobatan", lines=3, placeholder="Cara pengobatan...")
                            add_disease_symptoms = gr.Textbox(label="Gejala (pisahkan dengan koma)", placeholder="G01,G02,G03")
                            add_disease_btn = gr.Button("➕ Tambah Penyakit", variant="primary")
                            add_disease_result = gr.Markdown()
                            
                            add_disease_btn.click(
                                fn=system.add_disease,
                                inputs=[add_disease_id, add_disease_name, add_disease_info, add_disease_solution, add_disease_symptoms],
                                outputs=add_disease_result
                            )
                        
                        with gr.TabItem("✏️ Edit Penyakit"):
                            edit_disease_id = gr.Textbox(label="ID Penyakit", placeholder="P01")
                            edit_disease_name = gr.Textbox(label="Nama Penyakit (kosongkan jika tidak diubah)")
                            edit_disease_info = gr.Textbox(label="Informasi Penyakit (kosongkan jika tidak diubah)", lines=3)
                            edit_disease_solution = gr.Textbox(label="Solusi/Pengobatan (kosongkan jika tidak diubah)", lines=3)
                            edit_disease_symptoms = gr.Textbox(label="Gejala (pisahkan dengan koma, kosongkan jika tidak diubah)")
                            edit_disease_btn = gr.Button("✏️ Update Penyakit", variant="secondary")
                            edit_disease_result = gr.Markdown()
                            
                            edit_disease_btn.click(
                                fn=system.update_disease,
                                inputs=[edit_disease_id, edit_disease_name, edit_disease_info, edit_disease_solution, edit_disease_symptoms],
                                outputs=edit_disease_result
                            )
                        
                        with gr.TabItem("🗑️ Hapus Penyakit"):
                            delete_disease_id = gr.Textbox(label="ID Penyakit", placeholder="P01")
                            delete_disease_btn = gr.Button("🗑️ Hapus Penyakit", variant="stop")
                            delete_disease_result = gr.Markdown()
                            
                            delete_disease_btn.click(
                                fn=system.delete_disease,
                                inputs=[delete_disease_id],
                                outputs=delete_disease_result
                            )
                        
                        with gr.TabItem("📋 Lihat Semua Penyakit"):
                            view_diseases_btn = gr.Button("🔄 Refresh Daftar Penyakit")
                            diseases_list = gr.Markdown()
                            
                            view_diseases_btn.click(
                                fn=system.get_diseases_list,
                                outputs=diseases_list
                            )

                with gr.TabItem("📊 Statistik & Laporan"):
                    gr.Markdown("## Dashboard Admin")
                    
                    with gr.Row():
                        refresh_stats_btn = gr.Button("🔄 Refresh Statistik", variant="primary")
                        backup_btn = gr.Button("💾 Backup Data", variant="secondary")
                    
                    stats_display = gr.Markdown()
                    backup_result = gr.Markdown()
                    
                    def create_backup():
                        """Buat backup data"""
                        try:
                            import shutil
                            from datetime import datetime
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_files = [
                                (system.data_file, f"backup_data_{timestamp}.json"),
                                (system.users_file, f"backup_users_{timestamp}.json"),
                                (system.stats_file, f"backup_stats_{timestamp}.json")
                            ]
                            
                            backed_up = []
                            for source, backup_name in backup_files:
                                if os.path.exists(source):
                                    shutil.copy2(source, backup_name)
                                    backed_up.append(backup_name)
                            
                            if backed_up:
                                return f"✅ Backup berhasil dibuat:\n" + "\n".join([f"• {f}" for f in backed_up])
                            else:
                                return "❌ Tidak ada file yang di-backup!"
                                
                        except Exception as e:
                            return f"❌ Error saat backup: {str(e)}"
                    
                    refresh_stats_btn.click(
                        fn=system.get_consultation_stats,
                        outputs=stats_display
                    )
                    
                    backup_btn.click(
                        fn=create_backup,
                        outputs=backup_result
                    )

                with gr.TabItem("ℹ️ Bantuan & Info"):
                    gr.Markdown("""
                    # 🩺 Sistem Pakar Diagnosa Penyakit Telinga (Admin)
                    
                    ## Default Admin:
                    - **Username**: admin
                    - **Password**: admin123
                    
                    ## Default User:
                    - **Username**: user
                    - **Password**: user123
                    
                    *Silakan ganti password default setelah login pertama!*
                    """)

        # User Dashboard Components
        with gr.Column(visible=False) as user_dashboard:
            gr.Markdown("# 🩺 Sistem Pakar Diagnosa Penyakit Telinga")
            current_user_display = gr.Markdown()
            
            logout_btn = gr.Button("🚪 Logout", variant="secondary")
            
            with gr.Tabs():
                with gr.TabItem("🔍 Konsultasi Diagnosis"):
                    gr.Markdown("## Pilih gejala yang Anda rasakan:")
                    
                    inputs = []
                    
                    with gr.Column():
                        for code, symptom_name in system.symptoms.items():
                            symptom_check = gr.Checkbox(
                                label=f"{code}: {symptom_name}",
                                value=False
                            )
                            inputs.append(symptom_check)
                    
                    process_btn = gr.Button("🧮 Proses Diagnosis", variant="primary", size="lg")
                    
                    with gr.Column():
                        selected_output = gr.Markdown(label="Gejala Terpilih")
                        diagnosis_output = gr.Markdown(label="Hasil Diagnosis")
                        solution_output = gr.Markdown(label="Rekomendasi")
                        stats_output = gr.Markdown(label="Statistik")
                    
                    process_btn.click(
                        fn=system.process_diagnosis,
                        inputs=inputs,
                        outputs=[selected_output, diagnosis_output, solution_output, stats_output]
                    )

                with gr.TabItem("ℹ️ Bantuan & Info"):
                    gr.Markdown("""
                    # 🩺 Sistem Pakar Diagnosa Penyakit Telinga
                    
                    ## Panduan Penggunaan:
                    1. Pilih gejala yang Anda alami
                    2. Klik tombol "Proses Diagnosis"
                    3. Sistem akan memberikan hasil diagnosis dan rekomendasi
                    
                    ## Disclaimer:
                    Sistem ini hanya sebagai alat bantu diagnosis awal. Untuk penanganan yang tepat, selalu konsultasikan dengan dokter spesialis THT.
                    """)

        # Now define the login/logout functions that reference these components
        def handle_login(username, password):
            success, message, role = system.verify_login(username, password)
            if success:
                user_display = f"*Halo {'Admin' if role == 'admin' else ''} {username}*"
                if role == 'admin':
                    return {
                        login_screen: gr.update(visible=False),
                        admin_dashboard: gr.update(visible=True),
                        user_dashboard: gr.update(visible=False),
                        login_status: f"✅ {message} Selamat datang Admin {username}!",
                        current_user_display: user_display
                    }
                else:
                    return {
                        login_screen: gr.update(visible=False),
                        admin_dashboard: gr.update(visible=False),
                        user_dashboard: gr.update(visible=True),
                        login_status: f"✅ {message} Selamat datang {username}!",
                        current_user_display: user_display
                    }
            else:
                return {
                    login_status: f"❌ {message}"
                }

        def handle_logout():
            message = system.logout()
            return {
                admin_dashboard: gr.update(visible=False),
                user_dashboard: gr.update(visible=False),
                login_screen: gr.update(visible=True),
                login_status: f"✅ {message}",
                current_user_display: ""
            }

        # Connect the buttons to their functions
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_screen, admin_dashboard, user_dashboard, login_status, current_user_display]
        )

        logout_btn.click(
            fn=handle_logout,
            outputs=[admin_dashboard, user_dashboard, login_screen, login_status, current_user_display]
        )

    return demo

# Jalankan aplikasi
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )