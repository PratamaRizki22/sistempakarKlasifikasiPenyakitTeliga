#!/usr/bin/env python3
"""
Sistem Diagnosis Medis dengan Forward dan Backward Chaining
Implementasi berdasarkan diagram yang disediakan
"""

import json
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ConfidenceLevel(Enum):
    VERY_HIGH = 0.9
    HIGH = 0.7
    MEDIUM = 0.5
    LOW = 0.3
    VERY_LOW = 0.1

@dataclass
class Symptom:
    """Representasi gejala pasien"""
    name: str
    severity: float  # 0.0 - 1.0
    duration: int    # dalam hari
    is_present: bool = True

@dataclass
class RiskFactor:
    """Faktor risiko pasien"""
    name: str
    value: float     # 0.0 - 1.0
    weight: float    # bobot kepentingan

@dataclass
class MedicalHistory:
    """Riwayat medis pasien"""
    previous_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)

@dataclass
class Disease:
    """Representasi penyakit dalam knowledge base"""
    name: str
    symptoms: Dict[str, float]  # gejala -> certainty factor
    risk_factors: Dict[str, float]
    pathognomonic_signs: List[str]  # tanda patognomonik
    prevalence: float  # prevalensi dalam populasi
    demographics: Dict[str, Any]  # data demografis

@dataclass
class DiagnosisResult:
    """Hasil diagnosis"""
    disease: str
    confidence: float
    reasoning: List[str]
    recommendations: List[str]
    follow_up_questions: List[str]

class KnowledgeBase:
    """Knowledge Base untuk sistem diagnosis"""
    
    def __init__(self):
        self.diseases = {}
        self.demographic_weights = {}
        self.fuzzy_membership = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Inisialisasi data sampel - EXPAND DENGAN DATA LENGKAP"""
        
        # === ISI DENGAN DATA PENYAKIT ===
        sample_disease = Disease(
            name="Influenza",
            symptoms={
                "demam": 0.8,
                "batuk": 0.7,
                "sakit_kepala": 0.6,
                "kelelahan": 0.75,
                "nyeri_otot": 0.65
            },
            risk_factors={
                "musim_dingin": 0.6,
                "kontak_pasien": 0.8,
                "imunitas_rendah": 0.7
            },
            pathognomonic_signs=["onset_mendadak", "demam_tinggi"],
            prevalence=0.1,  # 10% populasi
            demographics={
                "age_risk": {"0-5": 0.8, "6-18": 0.6, "19-65": 0.4, "65+": 0.7},
                "gender_risk": {"male": 0.5, "female": 0.5}
            }
        )
        
        self.diseases["influenza"] = sample_disease
        
        # === ISI DENGAN DATA DEMOGRAFIS ===
        self.demographic_weights = {
            "age": 0.3,
            "gender": 0.2,
            "location": 0.25,
            "season": 0.25
        }
        
        # === ISI DENGAN FUZZY MEMBERSHIP FUNCTIONS ===
        self.fuzzy_membership = {
            "demam": {
                "normal": (36.0, 37.2),
                "ringan": (37.2, 38.0),
                "sedang": (38.0, 39.5),
                "tinggi": (39.5, 42.0)
            }
        }

class InferenceEngine:
    """Mesin inferensi dengan forward dan backward chaining"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.patient_data = {}
        self.diagnosis_results = []
    
    def forward_chaining(self, symptoms: List[Symptom], 
                        risk_factors: List[RiskFactor],
                        medical_history: MedicalHistory) -> List[Tuple[str, float]]:
        """Forward chaining untuk mendapatkan kandidat diagnosis"""
        
        candidates = []
        
        for disease_name, disease in self.kb.diseases.items():
            confidence = self._calculate_initial_confidence(
                disease, symptoms, risk_factors, medical_history
            )
            
            if confidence > 0.1:  # threshold minimum
                candidates.append((disease_name, confidence))
        
        # Sort berdasarkan confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _calculate_initial_confidence(self, disease: Disease, 
                                    symptoms: List[Symptom], 
                                    risk_factors: List[RiskFactor],
                                    medical_history: MedicalHistory) -> float:
        """Hitung confidence awal berdasarkan gejala dan faktor risiko"""
        
        symptom_score = 0.0
        symptom_count = 0
        
        # Evaluasi gejala
        for symptom in symptoms:
            if symptom.name in disease.symptoms and symptom.is_present:
                weight = disease.symptoms[symptom.name]
                severity_factor = symptom.severity
                symptom_score += weight * severity_factor
                symptom_count += 1
        
        # Evaluasi faktor risiko
        risk_score = 0.0
        risk_count = 0
        
        for risk in risk_factors:
            if risk.name in disease.risk_factors:
                weight = disease.risk_factors[risk.name]
                risk_score += weight * risk.value * risk.weight
                risk_count += 1
        
        # Kombinasi skor
        if symptom_count > 0:
            symptom_score /= symptom_count
        if risk_count > 0:
            risk_score /= risk_count
        
        # Weighted combination
        final_score = (symptom_score * 0.7) + (risk_score * 0.3)
        
        # Faktor prevalensi
        prevalence_factor = disease.prevalence
        
        return final_score * (1 + prevalence_factor)
    
    def has_tie(self, candidates: List[Tuple[str, float]], threshold: float = 0.1) -> bool:
        """Cek apakah ada tie dalam kandidat diagnosis"""
        if len(candidates) < 2:
            return False
        
        top_score = candidates[0][1]
        second_score = candidates[1][1]
        
        return abs(top_score - second_score) < threshold
    
    def backward_chaining(self, candidates: List[Tuple[str, float]], 
                         symptoms: List[Symptom]) -> DiagnosisResult:
        """Backward chaining untuk resolusi tie dan konfirmasi"""
        
        # === MODUL KONFIRMASI PATOGNOMONIK ===
        pathognomonic_scores = self._check_pathognomonic_signs(candidates, symptoms)
        
        # === ANALISIS TEMPORAL ===
        temporal_scores = self._analyze_temporal_patterns(candidates, symptoms)
        
        # === RESPONS PENGOBATAN ===
        treatment_scores = self._analyze_treatment_response(candidates)
        
        # === PEMERIKSAAN VIRTUAL ===
        virtual_exam_scores = self._virtual_examination(candidates)
        
        # === UPDATE CERTAINTY FACTORS ===
        final_scores = self._update_certainty_factors(
            candidates, pathognomonic_scores, temporal_scores,
            treatment_scores, virtual_exam_scores
        )
        
        # Pilih diagnosis dengan skor tertinggi
        best_diagnosis = max(final_scores, key=lambda x: x[1])
        
        return self._generate_diagnosis_result(best_diagnosis, candidates)
    
    def _check_pathognomonic_signs(self, candidates: List[Tuple[str, float]], 
                                  symptoms: List[Symptom]) -> Dict[str, float]:
        """Cek tanda patognomonik untuk setiap kandidat"""
        scores = {}
        
        for disease_name, _ in candidates:
            disease = self.kb.diseases[disease_name]
            pathognomonic_bonus = 0.0
            
            # === ISI DENGAN LOGIKA PATOGNOMONIK ===
            for sign in disease.pathognomonic_signs:
                if self._has_pathognomonic_sign(sign, symptoms):
                    pathognomonic_bonus += 0.3
            
            scores[disease_name] = pathognomonic_bonus
        
        return scores
    
    def _has_pathognomonic_sign(self, sign: str, symptoms: List[Symptom]) -> bool:
        """Cek apakah pasien memiliki tanda patognomonik tertentu"""
        # === ISI DENGAN LOGIKA DETEKSI TANDA PATOGNOMONIK ===
        sign_mappings = {
            "onset_mendadak": lambda s: any(sym.duration <= 2 and sym.severity > 0.7 for sym in s),
            "demam_tinggi": lambda s: any(sym.name == "demam" and sym.severity > 0.8 for sym in s)
        }
        
        if sign in sign_mappings:
            return sign_mappings[sign](symptoms)
        
        return False
    
    def _analyze_temporal_patterns(self, candidates: List[Tuple[str, float]], 
                                  symptoms: List[Symptom]) -> Dict[str, float]:
        """Analisis pola temporal gejala"""
        scores = {}
        
        for disease_name, _ in candidates:
            temporal_score = 0.0
            
            # === ISI DENGAN LOGIKA ANALISIS TEMPORAL ===
            # Contoh: onset cepat vs bertahap
            rapid_onset_symptoms = [s for s in symptoms if s.duration <= 3]
            gradual_onset_symptoms = [s for s in symptoms if s.duration > 7]
            
            if disease_name == "influenza" and len(rapid_onset_symptoms) > 2:
                temporal_score += 0.2
            
            scores[disease_name] = temporal_score
        
        return scores
    
    def _analyze_treatment_response(self, candidates: List[Tuple[str, float]]) -> Dict[str, float]:
        """Analisis respons terhadap pengobatan sebelumnya"""
        scores = {}
        
        for disease_name, _ in candidates:
            # === ISI DENGAN LOGIKA RESPONS PENGOBATAN ===
            treatment_score = 0.0
            
            # Placeholder untuk respons pengobatan
            scores[disease_name] = treatment_score
        
        return scores
    
    def _virtual_examination(self, candidates: List[Tuple[str, float]]) -> Dict[str, float]:
        """Pemeriksaan virtual tambahan"""
        scores = {}
        
        for disease_name, _ in candidates:
            # === ISI DENGAN LOGIKA PEMERIKSAAN VIRTUAL ===
            virtual_score = 0.0
            
            # Placeholder untuk pemeriksaan virtual
            scores[disease_name] = virtual_score
        
        return scores
    
    def _update_certainty_factors(self, candidates: List[Tuple[str, float]],
                                 pathognomonic: Dict[str, float],
                                 temporal: Dict[str, float],
                                 treatment: Dict[str, float],
                                 virtual: Dict[str, float]) -> List[Tuple[str, float]]:
        """Update certainty factors dengan informasi tambahan"""
        
        updated_scores = []
        
        for disease_name, initial_score in candidates:
            # Kombinasi semua faktor
            total_bonus = (pathognomonic.get(disease_name, 0) +
                          temporal.get(disease_name, 0) +
                          treatment.get(disease_name, 0) +
                          virtual.get(disease_name, 0))
            
            # Update dengan kombinasi non-linear
            updated_score = initial_score + (total_bonus * (1 - initial_score))
            updated_score = min(updated_score, 1.0)  # Cap at 1.0
            
            updated_scores.append((disease_name, updated_score))
        
        return sorted(updated_scores, key=lambda x: x[1], reverse=True)
    
    def _generate_diagnosis_result(self, best_diagnosis: Tuple[str, float],
                                  all_candidates: List[Tuple[str, float]]) -> DiagnosisResult:
        """Generate hasil diagnosis final"""
        
        disease_name, confidence = best_diagnosis
        disease = self.kb.diseases[disease_name]
        
        reasoning = [
            f"Diagnosis berdasarkan analisis gejala dan faktor risiko",
            f"Confidence score: {confidence:.2f}",
            f"Prevalensi penyakit: {disease.prevalence:.1%}"
        ]
        
        # === ISI DENGAN REKOMENDASI SPESIFIK ===
        recommendations = [
            "Istirahat yang cukup",
            "Konsumsi cairan yang banyak",
            "Konsultasi dengan dokter jika gejala memburuk"
        ]
        
        # === ISI DENGAN PERTANYAAN FOLLOW-UP ===
        follow_up_questions = [
            "Apakah Anda mengalami kesulitan bernapas?",
            "Sudah berapa lama gejala ini berlangsung?",
            "Apakah ada riwayat kontak dengan pasien serupa?"
        ]
        
        return DiagnosisResult(
            disease=disease_name,
            confidence=confidence,
            reasoning=reasoning,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions
        )

class MedicalDiagnosisSystem:
    """Sistem diagnosis medis utama"""
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.inference_engine = InferenceEngine(self.kb)
    
    def start_cli(self):
        """Mulai interface CLI"""
        print("=== SISTEM DIAGNOSIS MEDIS ===")
        print("Selamat datang di sistem diagnosis medis")
        print("-" * 40)
        
        while True:
            print("\nMenu:")
            print("1. Mulai Diagnosis")
            print("2. Lihat Knowledge Base")
            print("3. Keluar")
            
            choice = input("\nPilih menu (1-3): ").strip()
            
            if choice == '1':
                self.run_diagnosis()
            elif choice == '2':
                self.show_knowledge_base()
            elif choice == '3':
                print("Terima kasih telah menggunakan sistem diagnosis medis!")
                break
            else:
                print("Pilihan tidak valid!")
    
    def run_diagnosis(self):
        """Jalankan proses diagnosis"""
        print("\n=== PROSES DIAGNOSIS ===")
        
        # Input gejala
        symptoms = self.input_symptoms()
        
        # Input faktor risiko
        risk_factors = self.input_risk_factors()
        
        # Input riwayat medis
        medical_history = self.input_medical_history()
        
        print("\n--- MEMPROSES DIAGNOSIS ---")
        
        # Forward chaining
        candidates = self.inference_engine.forward_chaining(
            symptoms, risk_factors, medical_history
        )
        
        if not candidates:
            print("Tidak ditemukan diagnosis yang sesuai dengan gejala yang diberikan.")
            return
        
        print(f"Ditemukan {len(candidates)} kandidat diagnosis")
        
        # Cek tie
        if self.inference_engine.has_tie(candidates):
            print("Terdeteksi ambiguitas, menjalankan backward chaining...")
            result = self.inference_engine.backward_chaining(candidates, symptoms)
        else:
            # Langsung ke diagnosis
            best_candidate = candidates[0]
            result = self.inference_engine._generate_diagnosis_result(
                best_candidate, candidates
            )
        
        # Tampilkan hasil
        self.display_diagnosis_result(result)
    
    def input_symptoms(self) -> List[Symptom]:
        """Input gejala dari pengguna"""
        symptoms = []
        
        print("\n--- INPUT GEJALA ---")
        print("Masukkan gejala yang dialami (ketik 'selesai' untuk mengakhiri)")
        
        # === ISI DENGAN DAFTAR GEJALA LENGKAP ===
        available_symptoms = ["demam", "batuk", "sakit_kepala", "kelelahan", "nyeri_otot"]
        
        while True:
            print(f"\nGejala tersedia: {', '.join(available_symptoms)}")
            symptom_name = input("Nama gejala: ").strip().lower()
            
            if symptom_name == 'selesai':
                break
            
            if symptom_name not in available_symptoms:
                print("Gejala tidak dikenali!")
                continue
            
            try:
                severity = float(input("Tingkat keparahan (0.0-1.0): "))
                duration = int(input("Durasi dalam hari: "))
                
                symptoms.append(Symptom(
                    name=symptom_name,
                    severity=max(0.0, min(1.0, severity)),
                    duration=max(1, duration)
                ))
                
                print(f"Gejala '{symptom_name}' berhasil ditambahkan")
                
            except ValueError:
                print("Input tidak valid!")
        
        return symptoms
    
    def input_risk_factors(self) -> List[RiskFactor]:
        """Input faktor risiko"""
        risk_factors = []
        
        print("\n--- INPUT FAKTOR RISIKO ---")
        
        # === ISI DENGAN DAFTAR FAKTOR RISIKO LENGKAP ===
        available_risks = {
            "musim_dingin": "Sedang dalam musim dingin",
            "kontak_pasien": "Kontak dengan pasien sakit",
            "imunitas_rendah": "Sistem imun sedang lemah"
        }
        
        for risk_name, description in available_risks.items():
            response = input(f"{description}? (y/n): ").strip().lower()
            
            if response == 'y':
                try:
                    value = float(input(f"Tingkat risiko {risk_name} (0.0-1.0): "))
                    risk_factors.append(RiskFactor(
                        name=risk_name,
                        value=max(0.0, min(1.0, value)),
                        weight=0.5  # default weight
                    ))
                except ValueError:
                    print("Input tidak valid, menggunakan nilai default 0.5")
                    risk_factors.append(RiskFactor(
                        name=risk_name,
                        value=0.5,
                        weight=0.5
                    ))
        
        return risk_factors
    
    def input_medical_history(self) -> MedicalHistory:
        """Input riwayat medis"""
        print("\n--- INPUT RIWAYAT MEDIS ---")
        
        # === ISI DENGAN INPUT RIWAYAT MEDIS LENGKAP ===
        previous_conditions = input("Penyakit sebelumnya (pisahkan dengan koma): ").strip()
        medications = input("Obat yang sedang dikonsumsi (pisahkan dengan koma): ").strip()
        allergies = input("Alergi (pisahkan dengan koma): ").strip()
        family_history = input("Riwayat keluarga (pisahkan dengan koma): ").strip()
        
        return MedicalHistory(
            previous_conditions=[c.strip() for c in previous_conditions.split(',') if c.strip()],
            medications=[m.strip() for m in medications.split(',') if m.strip()],
            allergies=[a.strip() for a in allergies.split(',') if a.strip()],
            family_history=[f.strip() for f in family_history.split(',') if f.strip()]
        )
    
    def display_diagnosis_result(self, result: DiagnosisResult):
        """Tampilkan hasil diagnosis"""
        print("\n" + "="*50)
        print("HASIL DIAGNOSIS")
        print("="*50)
        
        print(f"Diagnosis: {result.disease.upper()}")
        print(f"Tingkat Kepercayaan: {result.confidence:.1%}")
        
        print(f"\nAlasan Diagnosis:")
        for reason in result.reasoning:
            print(f"• {reason}")
        
        print(f"\nRekomendasi:")
        for rec in result.recommendations:
            print(f"• {rec}")
        
        print(f"\nPertanyaan Lanjutan:")
        for question in result.follow_up_questions:
            print(f"• {question}")
        
        print("="*50)
    
    def show_knowledge_base(self):
        """Tampilkan isi knowledge base"""
        print("\n=== KNOWLEDGE BASE ===")
        
        print(f"Jumlah penyakit dalam database: {len(self.kb.diseases)}")
        
        for disease_name, disease in self.kb.diseases.items():
            print(f"\nPenyakit: {disease_name}")
            print(f"Prevalensi: {disease.prevalence:.1%}")
            print(f"Gejala: {list(disease.symptoms.keys())}")
            print(f"Faktor Risiko: {list(disease.risk_factors.keys())}")
            print(f"Tanda Patognomonik: {disease.pathognomonic_signs}")

def main():
    """Fungsi utama"""
    system = MedicalDiagnosisSystem()
    system.start_cli()

if __name__ == "__main__":
    main()