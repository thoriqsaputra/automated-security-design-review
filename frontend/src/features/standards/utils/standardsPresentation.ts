import type {
  DiagramRequirement,
  ParameterParent,
} from '../../../api/standards';

export const parameterPageSizeOptions = [5, 10, 20, 50, 100];

export function flattenParameterChildren(parameters: ParameterParent[]) {
  return parameters.flatMap((parent) => parent.children);
}

export function countDiagramRequirements(diagramRequirements: DiagramRequirement[]) {
  return diagramRequirements.length;
}
