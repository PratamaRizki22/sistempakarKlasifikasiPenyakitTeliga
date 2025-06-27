import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import hashlib
import threading
import time
from datetime import datetime

class EarDiagnosisSystem:
    def __init__(self):
        self.data_dir = "data"
        self.data_file = os.path.join(self.data_dir, "ear_diagnosis_data.json")
        self.stats_file = os.path.join(self.data_dir, "consultation_stats.json")
        self.stats_lock = threading.Lock() 
        self.last_save_time = time.time()
        self.save_interval = 5 

        self.severity_multipliers = {
            "tidak_parah": 0.3,
            "lumayan_parah": 0.6,  
            "parah": 0.85,           
            "sangat_parah": 1.0     
        }

        self.severity_labels = {
            "tidak_parah": "😊 Tidak Parah",
            "lumayan_parah": "😐 Lumayan Parah", 
            "parah": "😰 Parah",
            "sangat_parah": "😵 Sangat Parah"
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.load_data()
        self.consultation_count = 0
        self.disease_stats = {}
        self.load_stats()

    def load_data(self):
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
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    self.consultation_count = stats.get('consultation_count', 0)
                    self.disease_stats = stats.get('disease_stats', {})
            except Exception as e:
                print(f"Error loading stats: {e}")

    def save_stats(self):
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

            self.symptoms = {
                'G01': 'Gatal pada liang telinga',
                'G02': 'Sakit, terutama saat telinga disentuh atau ditarik',
                'G03': 'Keluar cairan bening pada telinga',
                'G04': 'Keluar cairan berwarna kuning atau bening dan berbau',
                'G05': 'Gangguan pendengaran (Pendengaran menurun)',
                'G06': 'Telinga terasa penuh atau tersumbat',
                'G07': 'Demam',
                'G08': 'Muncul benjolan dileher atau sekitar telinga',
                'G09': 'Vertigo dan pusing',
                'G10': 'Telinga berdenging',
                'G11': 'Nyeri Telinga',
                'G12': 'Demam disertai pilek',

            }

            self.diseases = {
                'P01': {
                    'name': 'Otitis Eksterna',
                    'symptoms': {
                        'G01': 0.8, 'G02': 0.7, 'G03': 0.6, 'G04': 0.5, 
                        'G05': 0.4, 'G06': 0.5, 'G07': 0.7, 'G11': 0.7
                    },
                    'info': 'Kombinasi gejala yang mengarah pada kemungkinan infeksi telinga luar atau peradangan luas.',
                    'solution': 'Jaga telinga tetap kering, hindari mengorek telinga. Segera konsultasikan dengan dokter THT untuk mendapatkan resep obat tetes atau antibiotik yang sesuai.',
                    'severity': 'Tinggi',
                    'duration': '1-2 minggu'
                },
                'P02': {
                    'name': 'Otitis Media',
                    'symptoms': {
                        'G04': 0.8, 'G05': 0.6, 'G06': 0.4, 'G07': 0.7, 
                        'G08': 0.6, 'G10': 0.4, 'G12': 0.6
                    },
                    'info': 'Gejala-gejala ini sering dikaitkan dengan infeksi pada telinga bagian tengah (Otitis Media), terutama jika ada demam dan pilek.',
                    'solution': 'Dibutuhkan pemeriksaan oleh dokter untuk konfirmasi. Penanganan bisa meliputi antibiotik dan pereda nyeri. Kompres hangat dapat membantu meringankan nyeri.',
                    'severity': 'Tinggi',
                    'duration': '5-10 hari'
                },
                'P03': {
                    'name': 'Gendang telinga pecah',
                    'symptoms': {
                        'G05': 0.6, 'G09': 0.5, 'G10': 0.6, 'G11': 0.7
                    },
                    'info': 'Kombinasi gangguan pendengaran, vertigo, dan telinga berdenging sering terkait dengan masalah pada telinga bagian dalam.',
                    'solution': 'Hindari gerakan kepala yang tiba-tiba. Konsultasikan dengan dokter untuk evaluasi fungsi pendengaran dan keseimbangan.',
                    'severity': 'Sedang',
                    'duration': 'Bervariasi'
                },
                'P04': {
                    'name': 'kolesteatoma',
                    'symptoms': {
                        'G03': 0.6, 'G04': 0.8, 'G05': 0.5, 'G06': 0.7, 
                        'G10': 0.6, 'G11': 0.4
                    },
                    'info': 'Pola gejala yang kompleks melibatkan infeksi (cairan), nyeri, sumbatan, dan gangguan pendengaran.',
                    'solution': 'Kondisi ini memerlukan evaluasi medis yang cermat. Jangan tunda untuk mengunjungi dokter THT untuk diagnosis yang akurat dan penanganan yang komprehensif.',
                    'severity': 'Tinggi',
                    'duration': 'Bervariasi'
                },
                'P05': {
                    'name': 'Presbikusis',
                    'symptoms': {
                        'G04': 0.5, 'G05': 0.6, 'G10': 0.8
                    },
                    'info': 'Gejala yang spesifik pada keluarnya cairan berbau, gangguan dengar, dan telinga berdenging. Bisa menandakan infeksi kronis.',
                    'solution': 'Sangat penting untuk diperiksakan ke dokter untuk mencegah komplikasi. Mungkin diperlukan pembersihan telinga profesional dan antibiotik.',
                    'severity': 'Sedang',
                    'duration': 'Bisa lama'
                }
            }

            self.save_data()

    def get_symptoms_list(self):
        if not self.symptoms:
            return "❌ Tidak ada gejala yang tersedia."
        
        result = "# 📋 Daftar Gejala Telinga\n\n"
        result += "Berikut adalah gejala-gejala yang dapat membantu dalam diagnosis penyakit telinga:\n\n"
        
        pain_symptoms = []
        hearing_symptoms = []
        discharge_symptoms = []
        balance_symptoms = []
        other_symptoms = []
        
        for code, desc in self.symptoms.items():
            if 'nyeri' in desc.lower():
                pain_symptoms.append((code, desc))
            elif 'pendengaran' in desc.lower() or 'berdenging' in desc.lower():
                hearing_symptoms.append((code, desc))
            elif 'cairan' in desc.lower() or 'bau' in desc.lower():
                discharge_symptoms.append((code, desc))
            elif 'pusing' in desc.lower() or 'keseimbangan' in desc.lower() or 'vertigo' in desc.lower():
                balance_symptoms.append((code, desc))
            else:
                other_symptoms.append((code, desc))
        
        if pain_symptoms:
            result += "## 🔥 Gejala Nyeri\n"
            for code, desc in pain_symptoms:
                result += f"- **{code}**: {desc}\n"
            result += "\n"
        
        if hearing_symptoms:
            result += "## 👂 Gejala Pendengaran\n"
            for code, desc in hearing_symptoms:
                result += f"- **{code}**: {desc}\n"
            result += "\n"
        
        if discharge_symptoms:
            result += "## 💧 Gejala Keluarnya Cairan\n"
            for code, desc in discharge_symptoms:
                result += f"- **{code}**: {desc}\n"
            result += "\n"
        
        if balance_symptoms:
            result += "## ⚖️ Gejala Keseimbangan\n"
            for code, desc in balance_symptoms:
                result += f"- **{code}**: {desc}\n"
            result += "\n"
        
        if other_symptoms:
            result += "## 🔹 Gejala Lainnya\n"
            for code, desc in other_symptoms:
                result += f"- **{code}**: {desc}\n"
            result += "\n"
        
        result += "---\n**💡 Tips**: Pilih semua gejala yang Anda rasakan untuk mendapatkan diagnosis yang lebih akurat."
        return result

    def get_diseases_list(self):
        if not self.diseases:
            return "❌ Tidak ada penyakit yang tersedia."
        
        result = "# 🏥 Daftar Penyakit Telinga\n\n"
        result += "Sistem ini dapat mendiagnosis berbagai penyakit telinga berdasarkan gejala yang Anda alami:\n\n"
        
        for code, disease in self.diseases.items():
            severity_emoji = "🔴" if disease.get('severity') == 'Tinggi' else "🟡" if disease.get('severity') == 'Sedang' else "🟢"
            
            result += f"## {severity_emoji} {code}: {disease['name']}\n\n"
            result += f"**📖 Deskripsi**: {disease['info']}\n\n"
            result += f"**⚠️ Tingkat Keparahan**: {disease.get('severity', 'Tidak diketahui')}\n\n"
            result += f"**⏱️ Durasi Biasanya**: {disease.get('duration', 'Bervariasi')}\n\n"
            
            result += f"**🎯 Gejala Terkait**: "
            symptom_names = []
            for symptom_code in disease['symptoms']:
                if symptom_code in self.symptoms:
                    symptom_names.append(f"{symptom_code} ({self.symptoms[symptom_code]})")
            result += ", ".join(symptom_names) + "\n\n"
            
            result += f"**💊 Rekomendasi Penanganan**: {disease['solution']}\n\n"
            result += "---\n\n"
        
        result += "**⚠️ Disclaimer**: Informasi ini hanya untuk referensi. Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis dan penanganan yang tepat."
        return result

    def get_consultation_stats(self):
        result = f"# 📊 Statistik Konsultasi Sistem\n\n"
        
        result += f"## 📈 Statistik Umum\n"
        result += f"- **Total Konsultasi**: {self.consultation_count:,} kali\n"
        result += f"- **Jumlah Penyakit**: {len(self.diseases)} jenis\n"
        result += f"- **Jumlah Gejala**: {len(self.symptoms)} gejala\n"
        result += f"- **Terakhir Diperbarui**: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}\n\n"
        
        if self.disease_stats:
            result += "## 🏆 Diagnosa Terpopuler\n"
            sorted_stats = sorted(self.disease_stats.items(), key=lambda x: x[1], reverse=True)
            
            total_diagnoses = sum(self.disease_stats.values())
            
            for i, (disease, count) in enumerate(sorted_stats[:5], 1):
                percentage = (count / total_diagnoses) * 100 if total_diagnoses > 0 else 0
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                result += f"{medal} **{disease}**: {count} kali ({percentage:.1f}%)\n"
            result += "\n"
        else:
            result += "## 🏆 Diagnosa Terpopuler\n"
            result += "Belum ada data konsultasi yang tersimpan.\n\n"
        
        if self.consultation_count > 0:
            result += "## 🎯 Informasi Sistem\n"
            result += f"- **Rata-rata gejala per konsultasi**: Bervariasi\n"
            result += f"- **Sistem aktif sejak**: Instalasi pertama\n"
            result += f"- **Status sistem**: ✅ Berjalan normal\n\n"
        
        result += "---\n**💡 Catatan**: Statistik ini membantu meningkatkan akurasi sistem diagnosis."
        
        return result

    def process_diagnosis(self, *args):

        symptom_mapping = [
            'G01', 'G02', 'G05', 'G06',    # Grup 1: 4 gejala
            'G08', 'G09', 'G11', 'G12',    # Grup 2: 4 gejala  
            'G03', 'G04', 'G07',           # Grup 3: 3 gejala
            'G10'                          # Grup 4: 1 gejala
        ]  # Total: 12 gejala sesuai dengan data
        
        selected_symptoms = {}
        
        try:
            for i in range(0, len(args), 2):
                if i + 1 < len(args): 
                    symptom_index = i // 2
                    if symptom_index < len(symptom_mapping):
                        symptom_code = symptom_mapping[symptom_index]
                        is_selected = bool(args[i]) 
                        severity = str(args[i + 1]) if args[i + 1] else "tidak_parah"
                        
                        valid_severities = ["tidak_parah", "lumayan_parah", "parah", "sangat_parah"]
                        if severity not in valid_severities:
                            severity = "tidak_parah"
                        
                        if is_selected:
                            selected_symptoms[symptom_code] = severity
                            
        except (IndexError, TypeError, ValueError) as e:
            print(f"ERROR: Argument parsing failed - {e}")
            error_result = "❌ **Terjadi kesalahan dalam memproses input!**\n\nSilakan refresh halaman dan coba lagi."
            return error_result, "", "", self.get_consultation_stats()

        print(f"DEBUG: Selected symptoms parsed: {selected_symptoms}")

        if not selected_symptoms:
            empty_result = "❌ **Silakan pilih minimal satu gejala terlebih dahulu!**\n\nPilih gejala yang Anda rasakan dari daftar di atas untuk mendapatkan diagnosis yang akurat."
            return empty_result, "", "", self.get_consultation_stats()

        inferred_facts, fired_rules = self.forward_chaining_inference(selected_symptoms)
        
        results = []
        for disease_code, disease in self.diseases.items():
            print(f"🔍 Analyzing disease: {disease_code} - {disease['name']}")
            
            if not isinstance(disease.get('symptoms'), dict):
                print(f"   ❌ SKIP: Invalid symptoms structure")
                continue

            cf_combined = self.calculate_combined_cf(
                disease['symptoms'], 
                selected_symptoms, 
                inferred_facts
            )
            
            matching_symptoms = [
                symptom for symptom in disease['symptoms'].keys() 
                if symptom in selected_symptoms
            ]
            
            print(f"   CF Combined: {cf_combined:.2f}%")
            print(f"   Matching symptoms: {len(matching_symptoms)}/{len(disease['symptoms'])}")
            
            # Threshold minimum untuk ditampilkan (40%)
            if cf_combined >= 40.0 and matching_symptoms:
                
                result = {
                    'code': disease_code,
                    'name': disease['name'],
                    'info': disease['info'],
                    'solution': disease['solution'],
                    'severity': disease.get('severity', 'Tidak diketahui'),
                    'duration': disease.get('duration', 'Bervariasi'),
                    'matching_symptoms': matching_symptoms,
                    'confidence': round(cf_combined, 1),
                    'total_symptoms': len(disease['symptoms']),
                    'matched_count': len(matching_symptoms),
                    'match_ratio': round((len(matching_symptoms) / len(disease['symptoms'])) * 100, 1),
                    'fired_rules': [rule for rule in fired_rules if rule.get('target_disease') == disease_code],
                    'risk_level': self.calculate_risk_level(cf_combined, disease.get('severity', 'Sedang'))
                }
                
                results.append(result)
                print(f"   ✅ ADDED: {disease['name']} - CF: {cf_combined:.1f}%")
            else:
                print(f"   ❌ SKIP: CF too low ({cf_combined:.1f}%) or no matches")

        print(f"DEBUG: Total valid results: {len(results)}")

        if results:
            for result in results:
                result['diagnosis_score'] = self.calculate_diagnosis_score(result)
            
            results.sort(key=lambda x: x['diagnosis_score'], reverse=True)


        selected_text, diagnosis_text, solution_text = self.format_results(selected_symptoms, results)
        updated_stats = self.get_consultation_stats()

        return selected_text, diagnosis_text, solution_text, updated_stats
    
    def update_consultation_stats(self, top_disease_name):
        with self.stats_lock:
            try:
                self.consultation_count += 1
                self.disease_stats[top_disease_name] = self.disease_stats.get(top_disease_name, 0) + 1
                
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self.save_stats_safely()
                    self.last_save_time = current_time
                    print(f"📊 Stats saved: {self.consultation_count} consultations")
                
            except Exception as e:
                print(f"ERROR updating stats: {e}")

    def save_stats_safely(self):

        stats_data = {
            'consultation_count': self.consultation_count,
            'disease_stats': self.disease_stats.copy(), 
            'last_updated': datetime.now().isoformat(),
            'version': '2.0' 
        }
        
        temp_file = self.stats_file + '.tmp'
        backup_file = self.stats_file + '.backup'
        
        try:
            if os.path.exists(self.stats_file):
                import shutil
                shutil.copy2(self.stats_file, backup_file)
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(temp_file):
                if os.name == 'nt': 
                    if os.path.exists(self.stats_file):
                        os.remove(self.stats_file)
                os.rename(temp_file, self.stats_file)
                
            return True
            
        except Exception as e:
            print(f"ERROR saving stats: {e}")
            
            if os.path.exists(backup_file):
                try:
                    import shutil
                    shutil.copy2(backup_file, self.stats_file)
                    print("📋 Stats restored from backup")
                except:
                    pass
            
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            return False

    def load_stats_safely(self):

        stats_files = [
            self.stats_file,
            self.stats_file + '.backup',
            self.stats_file + '.tmp'
        ]
        
        for stats_file in stats_files:
            if os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                        
                        if isinstance(stats, dict):
                            self.consultation_count = stats.get('consultation_count', 0)
                            self.disease_stats = stats.get('disease_stats', {})
                            
                            if not isinstance(self.consultation_count, int):
                                self.consultation_count = 0
                            if not isinstance(self.disease_stats, dict):
                                self.disease_stats = {}
                            
                            print(f"📊 Stats loaded from {stats_file}")
                            return True
                            
                except (json.JSONDecodeError, IOError) as e:
                    print(f"ERROR loading {stats_file}: {e}")
                    continue
        
        print("📊 Using default stats (no valid file found)")
        self.consultation_count = 0
        self.disease_stats = {}
        return False

    def forward_chaining_inference(self, selected_symptoms):
        working_memory = set(selected_symptoms.keys())
        fired_rules = []
        
        inference_rules = [
            {
                'id': 'R01',
                'name': 'Deteksi Pola Infeksi',
                'conditions': ['G01', 'G03', 'G07'], 
                'conclusion': 'INFECTION_PATTERN',
                'cf': 0.8,
                'target_disease': 'P02', 
                'description': 'Pola gejala menunjukkan kemungkinan infeksi telinga'
            },
            
            {
                'id': 'R02',
                'name': 'Deteksi Pola Sumbatan',
                'conditions': ['G11', 'G12', 'G10'],
                'conclusion': 'BLOCKAGE_PATTERN',
                'cf': 0.7,
                'target_disease': 'P03',
                'description': 'Pola gejala menunjukkan kemungkinan sumbatan telinga'
            },
            
            {
                'id': 'R03', 
                'name': 'Deteksi Pola Vertigo',
                'conditions': ['G9', 'G10', 'G5'],  
                'conclusion': 'VERTIGO_PATTERN',
                'cf': 0.9,
                'target_disease': 'P05', 
                'description': 'Pola gejala menunjukkan kemungkinan gangguan keseimbangan'
            },
            
            {
                'id': 'R04',
                'name': 'Deteksi Infeksi Eksternal',
                'conditions': ['G02', 'G04', 'G05'], 
                'conclusion': 'EXTERNAL_INFECTION_PATTERN', 
                'cf': 0.75,
                'target_disease': 'P01', 
                'description': 'Pola gejala menunjukkan kemungkinan infeksi telinga luar'
            },
            
            {
                'id': 'R05',
                'name': 'Deteksi Pola Tinnitus',
                'conditions': ['G15', 'G17'], 
                'conclusion': 'TINNITUS_PATTERN',
                'cf': 0.8,
                'target_disease': 'P04', 
                'description': 'Pola gejala menunjukkan kemungkinan tinnitus'
            },
            
            {
                'id': 'R06',
                'name': 'Deteksi Pola Tekanan',
                'conditions': ['G06', 'G13'], 
                'conclusion': 'PRESSURE_PATTERN',
                'cf': 0.6,
                'target_disease': 'P06', 
                'description': 'Pola gejala menunjukkan kemungkinan trauma tekanan'
            },
            
            {
                'id': 'R07',
                'name': 'Infeksi Kompleks',
                'conditions': ['INFECTION_PATTERN', 'G04'], 
                'conclusion': 'COMPLEX_INFECTION',
                'cf': 0.9,
                'target_disease': 'P02',
                'description': 'Infeksi dengan komplikasi'
            }
        ]
        
        max_iterations = 10 
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            new_facts_added = False
            
            print(f"   Forward Chaining Iteration {iteration}")
            print(f"   Working Memory: {working_memory}")
            
            for rule in inference_rules:
                if any(fired['id'] == rule['id'] for fired in fired_rules):
                    continue
                
                conditions_met = all(condition in working_memory for condition in rule['conditions'])
                
                if conditions_met and rule['conclusion'] not in working_memory:
                    working_memory.add(rule['conclusion'])
                    fired_rules.append(rule.copy())
                    new_facts_added = True
                    
                    print(f"   🔥 RULE FIRED: {rule['id']} - {rule['name']}")
                    print(f"      Conditions: {rule['conditions']} → {rule['conclusion']}")
            
            if not new_facts_added:
                print(f"   ✅ Forward chaining completed after {iteration} iterations")
                break
        
        print(f"   Final working memory: {working_memory}")
        print(f"   Total rules fired: {len(fired_rules)}")
        
        return working_memory, fired_rules


    def get_inference_explanation(self, fired_rules):

        if not fired_rules:
            return ""
        
        explanation = "### 🧠 Analisis Forward Chaining\n\n"
        explanation += "Sistem menggunakan forward chaining untuk menarik kesimpulan dari gejala:\n\n"
        
        for rule in fired_rules:
            explanation += f"**{rule['id']} - {rule['name']}**\n"
            explanation += f"- Kondisi: {', '.join(rule['conditions'])}\n"
            explanation += f"- Kesimpulan: {rule['conclusion']}\n"
            explanation += f"- CF: {rule['cf']}\n"
            explanation += f"- Deskripsi: {rule['description']}\n\n"
        
        return explanation


    def calculate_combined_cf(self, disease_symptoms, selected_symptoms, inferred_facts=None):

        if not disease_symptoms or not selected_symptoms:
            return 0.0
        
        cf_combined = 0.0
        processed_symptoms = []
        
        for symptom_code, base_cf in disease_symptoms.items():
            if symptom_code in selected_symptoms:
                severity = selected_symptoms[symptom_code]
                severity_multiplier = self.severity_multipliers.get(severity, 0.5)
                
                cf_symptom = float(base_cf) * float(severity_multiplier)

                print(f"   📊 {symptom_code}:")
                print(f"      Base CF: {base_cf} (type: {type(base_cf)})")
                print(f"      Severity: '{severity}' -> Multiplier: {severity_multiplier} (type: {type(severity_multiplier)})")
                print(f"      Calculation: {base_cf} × {severity_multiplier} = {cf_symptom}")
                print(f"      Manual check: {float(base_cf) * float(severity_multiplier)}")
                    
                if inferred_facts and symptom_code in inferred_facts:
                    cf_symptom = min(1.0, cf_symptom)
                    print(f"      With inference boost: {cf_symptom}")
                
                cf_previous = cf_combined
                if cf_combined == 0.0:
                    cf_combined = cf_symptom
                else:
                    # CF combining rule: CF1 + CF2 * (1 - CF1)
                    cf_combined = cf_combined + cf_symptom * (1 - cf_combined)

                print(f"      CF before combining: {cf_previous}")
                print(f"      CF after combining: {cf_combined}")
                print(f"      Combining formula: {cf_previous} + {cf_symptom} × (1 - {cf_previous}) = {cf_combined}")
                
                processed_symptoms.append({
                    'code': symptom_code,
                    'base_cf': base_cf,
                    'severity': severity,
                    'multiplier': severity_multiplier,
                    'final_cf': cf_symptom
                })
                
                print(f"      Symptom {symptom_code}: {base_cf} × {severity_multiplier} = {cf_symptom:.3f} (CF gejala)")
                print(f"      Combined CF so far: {cf_combined:.3f} (gabungan hingga gejala ini)")

        
        confidence_percentage = cf_combined * 100
        
        print(f"   FINAL CF: {cf_combined:.3f} = {confidence_percentage:.1f}%")
        
        return confidence_percentage


    def calculate_risk_level(self, cf_percentage, disease_severity):

        severity_weights = {
            'Tinggi': 1.0,
            'Sedang': 0.7, 
            'Ringan': 0.4
        }
        
        severity_weight = severity_weights.get(disease_severity, 0.5)
        risk_score = (cf_percentage / 100) * severity_weight
        
        if risk_score >= 0.8:
            return "🔴 RISK TINGGI"
        elif risk_score >= 0.6:
            return "🟡 RISK SEDANG"
        elif risk_score >= 0.3:
            return "🟢 RISK RENDAH"
        else:
            return "⚪ RISK MINIMAL"

    
    def calculate_diagnosis_score(self, result):

        return result['confidence']

    def format_enhanced_results(self, selected_symptoms, results, fired_rules):

        selected_text = "# 📋 Gejala yang Anda Pilih\n\n"
        selected_text += f"Anda telah memilih **{len(selected_symptoms)} gejala** dengan tingkat keparahan:\n\n"

        for i, (code, severity) in enumerate(selected_symptoms.items(), 1):
            symptom_name = self.symptoms.get(code, "Unknown")
            severity_label = self.severity_labels.get(severity, "😊 Tidak Parah")
            selected_text += f"{i}. **{code}**: {symptom_name}\n   📊 Tingkat: *{severity_label}*\n"

        selected_text += f"\n*Total gejala dipilih: {len(selected_symptoms)}*"

        if not results:
            return selected_text, "# 🤔 Hasil Diagnosis\n\n**Tidak ditemukan penyakit yang sesuai.**\n\nCoba pilih lebih banyak gejala atau konsultasi dengan dokter.", ""

        diagnosis_text = "# 🎯 Hasil Diagnosis\n\n"
        diagnosis_text += f"Berdasarkan {len(selected_symptoms)} gejala yang Anda pilih, sistem melakukan analisis forward chaining:\n\n"

        if fired_rules:
            diagnosis_text += self.get_inference_explanation(fired_rules)
            diagnosis_text += "\n"

        diagnosis_text += "## 🏆 Ranking Diagnosis:\n\n"

    def format_results(self, selected_symptoms, results):
        selected_text = "# 📋 Gejala yang Anda Pilih\n\n"
        selected_text += f"Anda telah memilih **{len(selected_symptoms)} gejala** berikut:\n\n"

        for i, code in enumerate(selected_symptoms, 1):
            symptom_name = self.symptoms.get(code, "Unknown")
            severity_label = self.severity_labels.get(selected_symptoms[code], "😊 Tidak Parah")
            selected_text += f"{i}. **{code}**: {symptom_name} — *{severity_label}*\n"

        selected_text += f"\n*Total gejala dipilih: {len(selected_symptoms)}*"

        if not results:
            return selected_text, "# 🤔 Hasil Diagnosis\n\n**Tidak ditemukan penyakit yang sesuai.**\n\nCoba pilih lebih banyak gejala atau konsultasi dengan dokter.", ""

        diagnosis_text = "# 🎯 Hasil Diagnosis\n\n"
        diagnosis_text += f"Berdasarkan {len(selected_symptoms)} gejala yang Anda pilih, berikut adalah hasil diagnosis yang mungkin:\n\n"

        for i, result in enumerate(results):
            rank_emoji = "🏆" if i == 0 else f"#{i+1}"
            confidence_emoji = "🔴" if result['confidence'] >= 80 else "🟡" if result['confidence'] >= 60 else "🟢"

            diagnosis_text += f"## {rank_emoji} {result['name']}\n"
            diagnosis_text += f"{confidence_emoji} **Confidence Factor (CF):** {result['confidence']}%\n"
            diagnosis_text += f"🎯 **Gejala Cocok:** {result['matched_count']} dari {result['total_symptoms']} gejala ({result['match_ratio']}%)\n"
            diagnosis_text += f"🔥 **Skor Gabungan:** {result.get('diagnosis_score', 0):.1f} / 100\n"
            diagnosis_text += f"⚠️ **Tingkat Keparahan Penyakit:** {result['severity']}\n\n"


            diagnosis_text += f"**📖 Deskripsi**: {result['info']}\n\n"
            diagnosis_text += f"**⚠️ Tingkat Keparahan**: {result['severity']}\n\n"
            diagnosis_text += f"**⏱️ Durasi Biasanya**: {result['duration']}\n\n"

            diagnosis_text += "### ✅ Gejala yang Cocok:\n"

            for code in result['matching_symptoms']:
                name = self.symptoms.get(code, "Unknown")
                base_cf = self.diseases[result['code']]['symptoms'].get(code, 0)
                user_severity = selected_symptoms.get(code, "tidak_parah")
                multiplier = self.severity_multipliers.get(user_severity, 0.5)
                cf_final = base_cf * multiplier
                level = "🔴 Tinggi" if cf_final >= 0.8 else "🟡 Sedang" if cf_final >= 0.5 else "⚪ Rendah"

                diagnosis_text += f"- **{code}**: {name}\n"
                diagnosis_text += f"  - CF Penyakit: {base_cf:.2f}\n"
                diagnosis_text += f"  - Tingkat: *{user_severity}* → Multiplier: {multiplier}\n"
                diagnosis_text += f"  - **CF Gejala:** {cf_final:.2f} ({level})\n"

            cf_values = [
                self.diseases[result['code']]['symptoms'][code] * self.severity_multipliers.get(selected_symptoms[code], 0.5)
                for code in result['matching_symptoms']
            ]

            if cf_values:
                cf_explanation = f"{cf_values[0]:.2f}"
                cf_combined = cf_values[0]
                for cf in cf_values[1:]:
                    cf_explanation += f" + {cf:.2f} × (1 - {cf_combined:.2f})"
                    cf_combined = cf_combined + cf * (1 - cf_combined)

                diagnosis_text += f"\n📊 **Perhitungan CF Gabungan:** {cf_explanation} = **{cf_combined * 100:.1f}%**\n"

            cf = result['confidence']
            match_ratio = result['match_ratio']
            severity = result['severity']
            fired_rules = result.get('fired_rules', [])

            w_cf = 0.4
            w_match = 0.3
            w_severity = 0.2
            w_rule = 0.1

            severity_scores = {'Tinggi': 100, 'Sedang': 70, 'Ringan': 40}
            severity_score = severity_scores.get(severity, 50)
            rule_score_raw = min(len(fired_rules) * 10, 50)

            diagnosis_score = (
                cf * w_cf +
                match_ratio * w_match +
                severity_score * w_severity +
                rule_score_raw * w_rule
            )

            diagnosis_text += f"\n🧮 **Skor Diagnosis Berdasarkan CF:** {cf:.1f}%\n"

            diagnosis_text += "\n---\n\n"



        primary_result = results[0]
        solution_text = f"# 💊 Rekomendasi Penanganan\n\n"
        solution_text += f"**Untuk diagnosis utama: {primary_result['name']}**\n\n"
        solution_text += f"{primary_result['solution']}\n\n"

        if primary_result['severity'] == 'Tinggi':
            solution_text += "🚨 **PERHATIAN KHUSUS**: Kondisi ini memerlukan penanganan segera!\n\n"
        elif primary_result['severity'] == 'Sedang':
            solution_text += "⚠️ **PERHATIAN**: Monitor perkembangan gejala dengan seksama.\n\n"

        solution_text += "## 📞 Kapan Harus ke Dokter?\n"
        solution_text += "Segera konsultasi dengan dokter THT jika:\n"
        solution_text += "- Gejala tidak membaik dalam 2-3 hari\n"
        solution_text += "- Nyeri semakin hebat\n"
        solution_text += "- Muncul demam tinggi\n"
        solution_text += "- Gangguan pendengaran bertambah parah\n\n"

        solution_text += "---\n\n"
        solution_text += "⚠️ **Disclaimer Penting**: Sistem ini hanya sebagai alat bantu diagnosis awal. "
        solution_text += "Untuk penanganan yang tepat dan akurat, selalu konsultasikan dengan dokter spesialis THT. "
        solution_text += "Jangan gunakan hasil ini sebagai pengganti konsultasi medis profesional."

        return selected_text, diagnosis_text, solution_text


def create_gradio_interface():
    with gr.Blocks(
        title="Sistem Pakar Diagnosa Penyakit Telinga", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-content {
            padding: 20px;
        }
        .symptom-checkbox {
            margin: 5px 0;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .result-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .gradio-container {
            max-width: 1200px !important;
        }
        .severity-radio {
            font-size: 12px !important;
        }
        .severity-radio label {
            font-size: 11px !important;
            margin: 2px 0 !important;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🩺 Sistem Pakar Diagnosa Penyakit Telinga
        
        ### Sistem diagnosis cerdas untuk membantu mengidentifikasi penyakit telinga berdasarkan gejala
        
        **✨ Fitur Unggulan:**
        - 🎯 Diagnosis akurat berdasarkan 18+ gejala
        - 🏥 Database 6+ penyakit telinga umum  
        - 📊 Statistik real-time (otomatis terupdate)
        - 💡 Rekomendasi penanganan komprehensif
        
        ---
        """)
        
        with gr.Tabs():
            with gr.TabItem("🔍 Konsultasi Diagnosis", elem_classes="tab-content"):
                gr.Markdown("""
                ## Langkah-langkah Konsultasi:
                1. **Pilih gejala** yang Anda rasakan dari daftar di bawah
                2. **Klik tombol "Analisis Gejala"** untuk memulai diagnosis
                3. **Lihat hasil** diagnosis dan rekomendasi penanganan
                4. **Konsultasi dengan dokter** untuk penanganan lebih lanjut
                
                ### Pilih semua gejala yang Anda rasakan:
                """)
                
                inputs = []
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔥 Gejala Nyeri & Sensitivitas")
                        for code in ['G01', 'G02', 'G05', 'G06']:
                            if code in system.symptoms:
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        symptom_check = gr.Checkbox(
                                            label=f"{code}: {system.symptoms[code]}",
                                            value=False,
                                            elem_classes="symptom-checkbox"
                                        )
                                    with gr.Column(scale=1):
                                        severity_radio = gr.Radio(
                                            choices=["tidak_parah", "lumayan_parah", "parah", "sangat_parah"],
                                            value="tidak_parah",
                                            label="Tingkat:",
                                            elem_classes="severity-radio",
                                            interactive=False 
                                        )
                                
                                symptom_check.change(
                                    fn=lambda checked: gr.update(interactive=checked),
                                    inputs=[symptom_check],
                                    outputs=[severity_radio]
                                )
                                
                                inputs.append(symptom_check)
                                inputs.append(severity_radio)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 👂 Masalah Pendengaran & Telinga")
                        for code in ['G08', 'G09', 'G11', 'G12']:
                            if code in system.symptoms:
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        symptom_check = gr.Checkbox(
                                            label=f"{code}: {system.symptoms[code]}",
                                            value=False,
                                            elem_classes="symptom-checkbox"
                                        )
                                    with gr.Column(scale=1):
                                        severity_radio = gr.Radio(
                                            choices=["tidak_parah", "lumayan_parah", "parah", "sangat_parah"],
                                            value="tidak_parah",
                                            label="Tingkat:",
                                            elem_classes="severity-radio",
                                            interactive=False 
                                        )
                                
                                symptom_check.change(
                                    fn=lambda checked: gr.update(interactive=checked),
                                    inputs=[symptom_check],
                                    outputs=[severity_radio]
                                )
                                
                                inputs.append(symptom_check)
                                inputs.append(severity_radio)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 💧 Cairan & Infeksi")
                        for code in ['G03', 'G04', 'G07']:
                            if code in system.symptoms:
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        symptom_check = gr.Checkbox(
                                            label=f"{code}: {system.symptoms[code]}",
                                            value=False,
                                            elem_classes="symptom-checkbox"
                                        )
                                    with gr.Column(scale=1):
                                        severity_radio = gr.Radio(
                                            choices=["tidak_parah", "lumayan_parah", "parah", "sangat_parah"],
                                            value="tidak_parah",
                                            label="Tingkat:",
                                            elem_classes="severity-radio",
                                            interactive=False 
                                        )
                                
                                symptom_check.change(
                                    fn=lambda checked: gr.update(interactive=checked), 
                                    inputs=[symptom_check],
                                    outputs=[severity_radio]
                                )
                                                                
                                inputs.append(symptom_check)
                                inputs.append(severity_radio)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚖️ Keseimbangan & Sistem")
                        for code in ['G10']:
                            if code in system.symptoms:
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        symptom_check = gr.Checkbox(
                                            label=f"{code}: {system.symptoms[code]}",
                                            value=False,
                                            elem_classes="symptom-checkbox"
                                        )
                                    with gr.Column(scale=1):
                                        severity_radio = gr.Radio(
                                            choices=["tidak_parah", "lumayan_parah", "parah", "sangat_parah"],
                                            value="tidak_parah",
                                            label="Tingkat:",
                                            elem_classes="severity-radio",
                                            interactive=False 
                                        )
                                
                                symptom_check.change(
                                    fn=lambda checked: gr.update(interactive=checked), 
                                    inputs=[symptom_check],
                                    outputs=[severity_radio]
                                )
                                
                                inputs.append(symptom_check)
                                inputs.append(severity_radio)
                
                with gr.Row():
                    process_btn = gr.Button(
                        "🧮 Analisis Gejala & Dapatkan Diagnosis", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button(
                        "🗑️ Reset Pilihan", 
                        variant="secondary", 
                        size="lg",
                        scale=1
                    )

                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=1):
                            selected_output = gr.Markdown(
                                label="Gejala Terpilih",
                                elem_classes="result-box"
                            )
                        with gr.Column(scale=1):
                            stats_output = gr.Markdown(
                                value=system.get_consultation_stats(),
                                label="Statistik Sistem",
                                elem_classes="result-box"
                            )
                    
                    diagnosis_output = gr.Markdown(
                        label="Hasil Diagnosis",
                        elem_classes="result-box"
                    )
                    solution_output = gr.Markdown(
                        label="Rekomendasi Penanganan",
                        elem_classes="result-box"
                    )
                
                process_btn.click(
                    fn=system.process_diagnosis,
                    inputs=inputs,
                    outputs=[selected_output, diagnosis_output, solution_output, stats_output]
                )
                
                def clear_all():
                    clear_values = []
                    for i in range(len(inputs)):
                        if i % 2 == 0: 
                            clear_values.append(False)
                        else: 
                            clear_values.append("tidak_parah")
                    
                    return clear_values + ["", "", "", system.get_consultation_stats()]
                
                clear_btn.click(
                    fn=clear_all,
                    outputs=inputs + [selected_output, diagnosis_output, solution_output, stats_output]
                )

    return demo

if __name__ == "__main__":
    system = EarDiagnosisSystem()
    
    print("🚀 Memulai Sistem Pakar Diagnosa Penyakit Telinga...")
    print(f"📊 Database: {len(system.diseases)} penyakit, {len(system.symptoms)} gejala")
    print(f"📈 Total konsultasi sebelumnya: {system.consultation_count}")
    print("🌐 Server akan berjalan di: http://localhost:7860")
    print("🔗 Link sharing akan tersedia setelah server aktif")
    print("=" * 60)
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )