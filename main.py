from pathlib import Path
import sys

ROOT = Path().cwd().resolve()
BASE_DIR = ROOT
MODELS_DIR   = BASE_DIR / 'models'
sys.path.append(str(MODELS_DIR))
sys.path.append(str(BASE_DIR))


# =============================================================================
# 🚀 Notebook principal de experimentación con TensorFlow/Keras
# =============================================================================

experiments = ["< EXP >"]  # Lista de experimentos a ejecutar
for experiment_name in experiments:
  # 1) Definir experimento
  #    El nombre debe coincidir con un archivo YAML en configs/experiments/
  exp_name = experiment_name

  # 2) Cargar configuración
  from utils.experiment.functions import load_config, load_experiment
  cfg = load_config(exp_name)
    
  # 3) ¿Single-split o K-Fold? | Repeticiones
  k = cfg["dataset"].get("k_folds")
  repeats = cfg["experiment"].get("repeats")

  for rep in range(repeats):

    if k is None or k <= 1:

        # —————————————— 3A) Flujo único ——————————————

        #  3A.1) Cargar experimento
        cfg, NNClass, params, dataset, train_data, val_data, test_idx = \
            load_experiment(exp_name, repeat_index=rep)

        #  3A.1.1) Revisar que 'rep' actual no haya sido previamente ejecutado
        #   Si ya existe classification_report.json  → SALTAR
        rep_report = BASE_DIR / cfg['experiment']['output_root'] / cfg['experiment']['output_subdir'] / "reports" / "classification_report.json"
        if rep_report.exists() == True:
          print(f"[SKIP] Rep: {rep} (single‐split) → ya existe classification_report.json.")
          continue

        #  3A.2) Instanciar y Entrenar
        model   = NNClass(cfg, **params)

        if rep == 0:
          #  3A.3 ) Mostrar resumen de la configuración
          from utils.misc.functions import print_exp_configuration
          print(f"\n✔️ Experimento «{cfg['experiment']['name']}» cargado con éxito.\n")
          print_exp_configuration(cfg)

          #  3A.4) Mostrar arquitectura del modelo
          print("\n📋 Arquitectura del modelo:")
          model.model.summary()

        print("\n"*5)
        print(f"\n🔄 Rep {rep+1}/{repeats}")
        #  4A.5) Entrenamiento (o retomar desde último checkpoint)
        history = model.fit(train_data, val_data)

        #  4A.6) Análisis resultados individual
        from utils.analysis.analysis import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer(
              model=model.model,
              history=history,
              test_data=test_idx,
              cfg=cfg,
              effects=dataset.get_effects("test"),
              repeat_index=rep,
              show_plots=False,
          )

        analyzer.classification_report()
        analyzer.effect_report()
        analyzer.confusion_matrix(normalize="true")
        model.cleanup_old_checkpoints()

    else:
        # —————————————— 3B) Flujo K-Fold ——————————————
        if rep == 0:
          #  3B.1) Cargar experimento con primer FOLD INDEX (fines informativos en consola)
          cfg, NNClass, params, _, _, _, _ = load_experiment(exp_name, fold_index=0, repeat_index=0)

          #  3B.2) Instanciar para obtener modelo y poder imprimir parámetros
          model   = NNClass(cfg, **params)

          #  3B.3 ) Mostrar resumen de la configuración
          from utils.misc.functions import print_exp_configuration
          print(f"\n✔️ Experimento «{cfg['experiment']['name']}» cargado con éxito.\n")
          print_exp_configuration(cfg)

          #  3B.4) Mostrar arquitectura del modelo
          print("\n📋 Arquitectura del modelo:")
          model.model.summary()

        #  3B.5) K-FOLD
        for fold in range(k):
            print("\n"*5)
            print(f"\n🔄 Rep {rep+1}/{repeats} | Fold {fold+1}/{k}")

            #  3B.5.1) Cargar experimento
            cfg, NNClass, params, dataset, train_data, val_data, test_idx = \
                load_experiment(exp_name, repeat_index=rep, fold_index=fold)

            #  3B.5.2) Revisar que 'rep' y 'fold' actual no hayan sido previamente ejecutados
            #   Si ya existe: /reports/classification_report.json  → SALTAR
            rep_report = BASE_DIR / cfg['experiment']['output_root'] / cfg['experiment']['output_subdir'] / "reports" / "classification_report.json"
            if rep_report.exists() == True:
              print(f"[SKIP] Rep: {rep} Fold: {fold}  → ya existe classification_report.json.")
              continue

            #  3B.5.4) Instanciar y Entrenar
            model   = NNClass(cfg, **params)
            history = model.fit(train_data, val_data)

            #  3B.5.4) Análisis resultados individual
            from utils.analysis.analysis import ExperimentAnalyzer
            analyzer = ExperimentAnalyzer(
                model=model.model,
                history=history,
                test_data=test_idx,
                cfg=cfg,
                repeat_index=rep,
                fold_index=fold,
                effects=dataset.get_effects("test"),
                show_plots=False,
            )

            # Guardar métricas en JSON
            analyzer.classification_report()
            analyzer.effect_report()
            analyzer.confusion_matrix(normalize="true")
            model.cleanup_old_checkpoints()

  print("\n"*4)
  print("="*15, f" | PROCESO {exp_name} FINALIZADO CORRECTAMENTE | ", "="*15)
  print("\n"*4)

# 4) Análisis Final
# from utils.analysis.analysis_rep import ExperimentRepAnalyzer
# analyzer = ExperimentRepAnalyzer(
#     cfg=load_config(exp_name)
# )

# analyzer.report_summary(confidence=0.95) # Resumen de métricas por repetición y fold
# analyzer.plot_evaluation(confidence=0.95) # Gráfica de evaluación (loss/accuracy con IC)

# analyzer.show_dashboard(confidence=0.95) # Dahsboard con todas las gráficas


# print("\n"*4)
# print("="*15, " | PROCESO FINALIZADO CORRECTAMENTE | ", "="*15)
# print("\n"*4)
