# import torch
# from Utils.plot import *
# import torch.distributed as dist

# def AllSky_eval(opt, model, LOADER_TEST, rank, world_size):
#     model.eval()  # Establecer el modelo en modo de evaluación
#     correct = 0
#     total = 0

#     predicciones = []
#     etiquetas = []

#     with torch.no_grad():  # No necesitamos calcular los gradientes durante la evaluación
#         for inputs, labels in LOADER_TEST:
#             # Mover los datos a la GPU
#             inputs, labels = inputs.to(f'cuda:{rank}'), labels.to(f'cuda:{rank}')

#             # Pasar los datos por el modelo
#             outputs = model(inputs)

#             # Obtener las predicciones
#             _, predicted = torch.max(outputs, 1)

#             # Guardar las predicciones y las etiquetas verdaderas
#             predicciones.append(predicted)
#             etiquetas.append(labels)

#             # Actualizar las métricas
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     # Sincronizar métricas entre todos los procesos (si es DDP)
#     if world_size > 1:
#         dist.reduce(torch.tensor(correct).to(f'cuda:{rank}'), dst=0, op=dist.ReduceOp.SUM)
#         dist.reduce(torch.tensor(total).to(f'cuda:{rank}'), dst=0, op=dist.ReduceOp.SUM)

#     # Promediar los resultados en rank 0
#     if rank == 0:
#         # Convertir las listas de predicciones y etiquetas a tensores
#         predicciones_tensor = torch.cat(predicciones, dim=0)
#         etiquetas_tensor = torch.cat(etiquetas, dim=0)

#         # Imprimir y guardar la matriz de confusión
#         plot_confusion_matrix(opt, predicciones_tensor, etiquetas_tensor)

#         precision = 100 * correct / total

#         # Imprimir la precisión del conjunto de prueba
#         print(f"Precisión en el conjunto de prueba: {precision:.2f}%")

#         return precision
#     else:
#         return None
