#!/usr/bin/env python3
"""
Industrial Quality Control LLaMA Instruction Dataset Generator
============================================================
Generates a domain-specific instruction fine-tuning dataset for LLaMA 3.2 (1B/3B)
covering four industrial materials and 20 defect classes:
  1. Steel Strip (NEU-DET): crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
  2. Glass Bottle (MVTec AD): broken_large, broken_small, contamination
  3. Wire Mesh Grid (MVTec AD): bent, broken, glue, metal_contamination, thread
  4. Ceramic Tile (MVTec AD): crack, cut, glue_strip, gray_stroke, oil, rough
"""

import json
import random
from pathlib import Path

random.seed(42)

# Domain Knowledge Base for the four inspection categories
DEFECT_KB = {
    "steel": {
        "material": "Hot/Cold-Rolled Carbon Steel Sheet & Strip (ASTM A1008/A568)",
        "defects": {
            "rolled-in_scale": {
                "severities": ["CRITICAL", "REJECT"],
                "assessments": [
                    "Embedded iron oxide scale disrupts grain continuity and induces catastrophic tearing during deep-drawing stamping.",
                    "High-temperature mill scale pressed into strip surface impairs coating adhesion and causes severe fracture risks.",
                    "Entrained primary scale creates macro-notches under high tensile deformation loads."
                ],
                "actions": [
                    "Quarantine master coil immediately. Inspect descaler header spray nozzles and check water pressure (>15 MPa) on Stands F1-F3.",
                    "Divert coil to non-critical commercial grade. Verify furnace oxygen atmosphere and descaling pump filters.",
                    "Stop rolling line. Clean descaling spray bar and inspect roughing stand roll surface finish."
                ]
            },
            "crazing": {
                "severities": ["CRITICAL", "REJECT"],
                "assessments": [
                    "Interconnected micro-crack network initiates rapid fatigue fracture under cyclic structural loads.",
                    "Surface thermal-stress crazing severely compromises tensile yield strength and burst pressure.",
                    "Micro-fracture matrix accelerates stress-corrosion cracking in corrosive environments."
                ],
                "actions": [
                    "Reject strip lot. Inspect continuous caster mold oscillation and verify primary cooling water uniformity.",
                    "Quarantine material for metallurgical cross-sectioning. Check secondary cooling zone spray patterns.",
                    "Scrap affected slab segment. Recalibrate mold level control and check tundish superheat temperature."
                ]
            },
            "inclusion": {
                "severities": ["MAJOR", "QUARANTINE"],
                "assessments": [
                    "Subsurface non-metallic alumina inclusions create localized stress concentration points during forming.",
                    "Entrapped slag particle inclusions degrade surface finish and induce localized galling on stamping dies.",
                    "Foreign metallurgical refractory inclusions trigger pinhole defects during automotive stamping."
                ],
                "actions": [
                    "Quarantine coil for ultrasonic cleanliness evaluation. Verify ladle shroud argon sealing.",
                    "Divert strip to secondary low-draw application. Audit tundish argon bubbling flow rates.",
                    "Conduct microscopic clean-steel analysis. Inspect submerged entry nozzle (SEN) clogging."
                ]
            },
            "pitted_surface": {
                "severities": ["MAJOR", "REWORK"],
                "assessments": [
                    "Surface micro-cavities from acid pickling over-exposure create galvanic corrosion pockets.",
                    "Localized pitting cavities destroy aesthetic coating uniformity and reduce effective strip thickness.",
                    "Aggressive electrochemical pitting reduces corrosion resistance threshold below automotive spec."
                ],
                "actions": [
                    "Divert batch to surface conditioning pickling line. Recalibrate hydrochloric acid bath temperatures.",
                    "Rework via controlled skin-pass temper rolling. Audit pickling inhibitor concentration levels.",
                    "Quarantine coil. Adjust pickling line speed and verify rinse bath pH neutrality."
                ]
            },
            "patches": {
                "severities": ["MODERATE", "REWORK"],
                "assessments": [
                    "Non-uniform surface oxidation patches cause galvanic potential variations during electro-galvanizing.",
                    "Discolored roll-mark patches indicate localized roll slippage and friction variation.",
                    "Surface oxide patch clusters impair paint primer wettability and electro-deposition adhesion."
                ],
                "actions": [
                    "Route coil through skin-pass temper mill for surface re-texturing. Clean work roll wipers.",
                    "Apply light abrasive conditioning brush. Inspect roll coolant emulsion lubrication concentration.",
                    "Release with concession for painted underbody parts only. Inspect bridle roll tension balance."
                ]
            },
            "scratches": {
                "severities": ["MODERATE", "REWORK"],
                "assessments": [
                    "Longitudinal mechanical score lines compromise zinc coating adhesion and aesthetic Class-A requirements.",
                    "Guide-table friction scratches induce premature micro-creasing under tension leveling.",
                    "Mechanical abrasive gouges disrupt surface passivation layer and invite atmospheric oxidation."
                ],
                "actions": [
                    "Inspect line guide roller bearings and deflector roll clearances. Polish deburring blades.",
                    "Rework by tension leveling with protective poly film. Clean coiler pinch roll entry guides.",
                    "Quarantine coil. Remove hardened burrs from coil accumulation looping pit guides."
                ]
            }
        }
    },
    "bottle": {
        "material": "PET / Borosilicate Glass Beverage Container (Food & Pharma Grade)",
        "defects": {
            "broken_large": {
                "severities": ["CRITICAL", "REJECT"],
                "assessments": [
                    "Catastrophic structural fracture destroys hermetic seal, risking total vacuum loss and glass particle shedding.",
                    "Severe macro-wall blowout eliminates pressure containment and creates acute shatter hazards on filling lines.",
                    "Extensive wall fracture breaches sterile barrier and induces downstream conveyor jamming."
                ],
                "actions": [
                    "Immediate pneumatic line ejection to scrap hopper. Stop blow-molder and check clamp pressure alignment.",
                    "Quarantine preceding 500-unit lot for vacuum decay testing. Inspect preform reheat oven lamps.",
                    "Halt packaging cell. Clean conveyor bed of micro-shards and recalibrate neck starwheel timing."
                ]
            },
            "broken_small": {
                "severities": ["MAJOR", "REJECT"],
                "assessments": [
                    "Localized neck finish chip compromises crown cap hermetic seal, inducing beverage leakage and oxidation.",
                    "Micro-fracture on closure sealing bead allows carbonation loss and microbial ingress.",
                    "Small rim fracture creates sharp edge hazard and risks capper head chuck misalignment."
                ],
                "actions": [
                    "Divert bottle to reject bin. Inspect transfer gripper fingers and mold takeout jaws.",
                    "Quarantine pallet lot for seal torque audit. Verify blow mold neck insert alignment.",
                    "Reject unit. Inspect capper spindle descent pressure and sorter starwheel clearance."
                ]
            },
            "contamination": {
                "severities": ["CRITICAL", "REJECT"],
                "assessments": [
                    "Foreign particulate matter adhering to inner cavity violates FDA/ISO 22000 hygiene safety standards.",
                    "Internal foreign particulate residue causes chemical leaching and batch beverage spoilage.",
                    "Spec particulate foreign body inside container risks customer safety and regulatory recall."
                ],
                "actions": [
                    "Eject contaminated container. Inspect ionizing de-ionizer air rinser filtration and nozzle pressure.",
                    "Quarantine current filling pallet. Clean HEPA air filter bank and inspect bottle washer jets.",
                    "Halt line. Verify cleanroom positive pressure envelope and swab rinse water supply line."
                ]
            }
        }
    },
    "grid": {
        "material": "Stamped Structural Wire Net / Sheet Metal Mesh",
        "defects": {
            "bent": {
                "severities": ["MAJOR", "REWORK"],
                "assessments": [
                    "Geometric strut curvature compromises mesh planar tolerance and reduces structural load shear resistance.",
                    "Plastic deformation on wire strut distorts aperture spacing and induces localized stress concentrations.",
                    "Strut bending exceeds geometric dimensional tolerance, risking mechanical interference in final housing."
                ],
                "actions": [
                    "Divert part to hydraulic straightening jig. Inspect stamping die clearance and guide rail clearance.",
                    "Route to rework station for dimensional re-gauging. Check automated wire feeder tensioning belts.",
                    "Adjust forming punch alignment. Inspect wire uncoiler leveling rollers for lateral binding."
                ]
            },
            "broken": {
                "severities": ["CRITICAL", "REJECT"],
                "assessments": [
                    "Severed wire strut destroys structural load-bearing continuity and causes rapid mesh unraveling under vibration.",
                    "Tensile strut fracture eliminates structural shear resistance and creates loose sharp wire hazards.",
                    "Complete wire node separation violates mechanical shear strength specifications."
                ],
                "actions": [
                    "Scrap component immediately. Inspect stamping shear blades for dulling or excessive clearance.",
                    "Quarantine batch. Examine spot-welding electrode current profile and replace worn welding tips.",
                    "Stop press. Inspect uncoiler wire tensile uniformity and calibrate punch stroke depth."
                ]
            },
            "glue": {
                "severities": ["MODERATE", "REWORK"],
                "assessments": [
                    "Adhesive resin overflow occludes mesh apertures, restricting required fluid and airflow volume.",
                    "Excessive adhesive droplet alters component weight balance and impairs downstream assembly flushness.",
                    "Polymer resin residue across mesh openings restricts acoustic and ventilation specifications."
                ],
                "actions": [
                    "Divert to ultrasonic solvent cleaning tank. Recalibrate automated glue dispensing valve shut-off pressure.",
                    "Rework with thermal heat-gun softening and scraping. Clean automated adhesive nozzle tips.",
                    "Adjust adhesive bead viscosity and decrease metering pump dispensing volume by 15%."
                ]
            },
            "metal_contamination": {
                "severities": ["MAJOR", "REWORK"],
                "assessments": [
                    "Specular metallic scrap shavings lodged in aperture risk electrical short circuits in downstream assembly.",
                    "Loose metallic solder chips create galvanic corrosion cells and abrasive wear on mating parts.",
                    "Conductive metal shavings across mesh struts present acute electrical isolation failure risks."
                ],
                "actions": [
                    "Pass through automated magnetic separator and high-pressure air de-burring tunnel.",
                    "Divert to ultrasonic cleaning wash. Inspect stamping scrap evacuation vacuum chute for clogging.",
                    "Rework with electrostatic dust-removal wand. Clean overhead transfer conveyor guides."
                ]
            },
            "thread": {
                "severities": ["MINOR", "REWORK"],
                "assessments": [
                    "Stray lint fibers draped across wire struts cause optical sensor false triggers and cosmetic rejection.",
                    "Non-structural fibrous string snagged in mesh aperture creates particulate shedding risks.",
                    "Organic thread lint trapped in wire net impairs aesthetic surface finish before powder coating."
                ],
                "actions": [
                    "Route through automated ionized air blow-off station to dislodge stray fibers.",
                    "Rework with manual lint vacuum wand. Audit operator cleanroom gowning and glove lint specifications.",
                    "Clean conveyor anti-static wipers. Inspect upstream packaging material for fiber shedding."
                ]
            }
        }
    },
    "tile": {
        "material": "Ceramic / Porcelain Flooring & Wall Tiles (ISO 10545)",
        "defects": {
            "crack": {
                "severities": ["CRITICAL", "REJECT"],
                "assessments": [
                    "Structural stress crack propagates under 1500N floor loading per ISO 10545-4, causing tile breakage.",
                    "Thermal cooling fracture penetrates tile thickness, destroying frost resistance and structural integrity.",
                    "Internal press fracture severely impairs modulus of rupture and creates acute floor safety hazards."
                ],
                "actions": [
                    "Pneumatically reject tile to cullet crusher. Inspect hydraulic press punch pressure balance.",
                    "Stop line. Verify Kiln Zone 3 rapid cooling ramp (<15°C/min) to avoid thermal shock.",
                    "Quarantine production run. Audit spray-dried clay moisture content and press punch de-airing."
                ]
            },
            "cut": {
                "severities": ["MAJOR", "REJECT"],
                "assessments": [
                    "Deep mechanical gouge penetrates glazed enamel surface, exposing porous ceramic biscuit to moisture.",
                    "Sharp tooling cut impairs abrasion resistance and creates noticeable aesthetic floor surface blemish.",
                    "Biscuit scoring cut weakens mechanical tile edge retention under transit packaging vibration."
                ],
                "actions": [
                    "Reject tile to scrap. Inspect transfer starwheels and conveyor chain guide alignment for sharp burrs.",
                    "Quarantine tile stack. Check sizing and squaring diamond wheel feed rates.",
                    "Adjust tile turnover mechanism dampers. Polish steel transfer conveyor rails."
                ]
            },
            "glue_strip": {
                "severities": ["MODERATE", "REWORK"],
                "assessments": [
                    "Adhesive bonding strip residue on tile face prevents proper grout mortar adhesion during installation.",
                    "Packaging hot-melt adhesive smear degrades glazed surface gloss and attracts transit dirt.",
                    "Polymer resin residue creates surface thickness variation exceeding automated sorting flatness specs."
                ],
                "actions": [
                    "Divert tile to rotary buffing solvent wash station. Recalibrate packaging adhesive applicator cutoff.",
                    "Rework with mechanical scraping wiper. Inspect stacker hot-melt glue gun temperature settings.",
                    "Clean packaging robot suction pads. Reduce hot-melt nozzle discharge volume by 10%."
                ]
            },
            "gray_stroke": {
                "severities": ["MINOR", "REWORK"],
                "assessments": [
                    "Metal-marking friction stroke from aluminum conveyor rail impairs decorative glaze color uniformity.",
                    "Surface graphite rub mark degrades Class-1 aesthetic finish per visual inspection standard.",
                    "Abrasion skid mark on polished glaze requires cosmetic surface buffing before cartoning."
                ],
                "actions": [
                    "Route tile through inline felt polishing wheel with cerium oxide slurry.",
                    "Rework via micro-abrasive rotary pad. Cover metal transfer rails with Teflon protective wear strips.",
                    "Clean conveyor alignment guides. Inspect rubber belt traction to prevent tile slippage."
                ]
            },
            "oil": {
                "severities": ["MAJOR", "REWORK"],
                "assessments": [
                    "Machine lubricant droplet penetrates unglazed tile pores, creating permanent dark staining after kiln firing.",
                    "Hydrocarbon oil contamination repels tile glaze slurry, forming unglazed surface pinhole voids.",
                    "Lubricant splatter creates slippery handling hazards and destroys tile adhesive bonding strength."
                ],
                "actions": [
                    "Divert tile for degreasing solvent wash before kiln entry. Inspect overhead hydraulic press seals.",
                    "Rework with chemical absorbent degreaser. Check pneumatic chain lubricator drip-feed rate.",
                    "Quarantine green tiles. Replace leaking overhead conveyor bearing seals immediately."
                ]
            },
            "rough": {
                "severities": ["MODERATE", "REWORK"],
                "assessments": [
                    "Surface glaze texture roughness exceeds Ra roughness specification, causing difficult cleaning maintenance.",
                    "Inadequate glaze flow during firing leaves orange-peel biscuit texture and uneven light reflectivity.",
                    "Coarse glaze granule agglomeration fails non-slip tactile and aesthetic grading specifications."
                ],
                "actions": [
                    "Route tile through diamond disc honing station for surface finish calibration.",
                    "Rework via rotary glaze polishing head. Check glaze ball mill grinding fineness (<0.5% on 45 micron sieve).",
                    "Adjust kiln peak firing temperature (+10°C) to promote smooth glaze melting flow."
                ]
            }
        }
    }
}

INSTRUCTION_PROMPT = (
    "You are an expert Industrial Quality Control (QC) Engineer. "
    "Evaluate the visual defect detected by the computer vision inspection system and provide: "
    "(1) Severity level (CRITICAL, MAJOR, MODERATE, MINOR), "
    "(2) Technical root cause engineering assessment (max 1 sentence), and "
    "(3) Immediate manufacturing corrective action (max 1 sentence)."
)

def generate_samples(num_samples=1200):
    samples = []
    categories = list(DEFECT_KB.keys())
    
    for i in range(num_samples):
        cat = random.choice(categories)
        cat_info = DEFECT_KB[cat]
        material_str = cat_info["material"]
        
        defect_name = random.choice(list(cat_info["defects"].keys()))
        defect_data = cat_info["defects"][defect_name]
        
        # Simulated realistic detection telemetry
        area_pct = round(random.uniform(0.2, 12.5), 1)
        conf_pct = random.randint(72, 99)
        
        severity = random.choice(defect_data["severities"])
        assessment = random.choice(defect_data["assessments"])
        action = random.choice(defect_data["actions"])
        
        verdict = "REJECT" if severity == "CRITICAL" else ("QUARANTINE" if severity == "MAJOR" else "REWORK")
        
        input_text = (
            f"Part: {cat.upper()} ({material_str}). "
            f"Defect: {defect_name} (area: {area_pct}%, confidence: {conf_pct}%). "
            f"Preliminary Rule Verdict: {verdict}."
        )
        
        output_text = (
            f"- Severity: {severity}\n"
            f"- Assessment: {assessment}\n"
            f"- Action: {action}"
        )
        
        sample = {
            "instruction": INSTRUCTION_PROMPT,
            "input": input_text,
            "output": output_text,
            "category": cat,
            "defect": defect_name,
            "severity": severity
        }
        samples.append(sample)
        
    return samples

def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = out_dir / "industrial_qc_llama_dataset.json"
    jsonl_path = out_dir / "industrial_qc_llama_dataset.jsonl"
    
    print("=" * 70)
    print("Industrial Quality Control LLaMA Instruction Dataset Generator")
    print(f"Target Categories: {list(DEFECT_KB.keys())}")
    print(f"Total Unique Defect Classes: {sum(len(v['defects']) for v in DEFECT_KB.values())}")
    print("=" * 70)
    
    samples = generate_samples(num_samples=1200)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
        
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(samples)} QC instruction samples.")
    print(f"  JSON format:  {json_path}")
    print(f"  JSONL format: {jsonl_path}")
    
    print("\n--- SAMPLE PREVIEW ---")
    s0 = samples[0]
    print(f"INPUT:\n{s0['input']}\n")
    print(f"OUTPUT:\n{s0['output']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
