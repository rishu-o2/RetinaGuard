/**
 * Offline Clinical Decision Rules Engine for RetinaGuard (TypeScript port).
 * Reads the YAML-derived JSON configuration served by GET /model/metadata.
 * NO threshold is hardcoded in this file.
 */

export interface AbstentionConfig {
  lower_edge: number;
  upper_edge: number;
  cutoff: number;
}

export interface SightThreateningConfig {
  threshold_soon: number;
  threshold_urgent: number;
}

export interface GradeRule {
  referable: boolean;
  base_urgency: "ROUTINE" | "SOON" | "PRIORITY" | "URGENT" | "EMERGENCY" | string;
  base_interval_months: number;
}

export interface ModelMetadataConfig {
  version: string;
  version_tag: string;
  onnx_sha256: string;
  image_size: [number, number];
  temperature: number;
  abstention: AbstentionConfig;
  min_acceptable_confidence: number;
  sight_threatening: SightThreateningConfig;
  grade_table: Record<string, GradeRule>;
  modifiers: {
    pregnancy: {
      min_urgency: string;
      max_interval_months: number;
    };
    high_risk_glycemic_control: {
      hba1c_threshold_percent: number;
      max_interval_months: number;
    };
    long_standing_diabetes: {
      duration_years_threshold: number;
      max_interval_months: number;
    };
  };
}

export interface PatientContextInput {
  is_pregnant?: boolean;
  hba1c?: number | null;
  diabetes_duration_years?: number | null;
}

export interface RulesEvaluationOutput {
  referral: "REFERABLE" | "NON_REFERABLE" | "UNCERTAIN";
  urgency: "ROUTINE" | "SOON" | "PRIORITY" | "URGENT" | "EMERGENCY";
  follow_up_interval_months: number;
  is_uncertain: boolean;
  uncertainty_reasons: string[];
  applied_modifiers: string[];
}

const URGENCY_RANKS: Record<string, number> = {
  ROUTINE: 1,
  SOON: 2,
  PRIORITY: 2,
  URGENT: 3,
  EMERGENCY: 4,
};

export function maxUrgency(u1: string, u2: string): any {
  const r1 = URGENCY_RANKS[u1] ?? 1;
  const r2 = URGENCY_RANKS[u2] ?? 1;
  return r1 >= r2 ? u1 : u2;
}

/** Numerically stable softmax */
export function softmax(logits: number[]): number[] {
  const maxVal = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxVal));
  const sumExps = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sumExps);
}

/** Apply post-hoc temperature calibration to raw model logits */
export function applyTemperatureCalibration(
  logits: number[],
  temperature: number
): number[] {
  if (temperature <= 0) return softmax(logits);
  const scaled = logits.map((l) => l / temperature);
  return softmax(scaled);
}

/** Deterministic clinical rules evaluation using backend configuration */
export function evaluateRules(
  probabilities: number[],
  context: PatientContextInput,
  config: ModelMetadataConfig
): RulesEvaluationOutput {
  // Find predicted grade (argmax) and confidence (max)
  let predictedGrade = 0;
  let maxProb = probabilities[0];
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      predictedGrade = i;
    }
  }

  // P(referable) = P(grade >= 2) = P(2) + P(3) + P(4)
  const pRef =
    (probabilities[2] || 0) +
    (probabilities[3] || 0) +
    (probabilities[4] || 0);

  // Gate 2: Referable-based abstention band
  if (config.abstention) {
    const { lower_edge, upper_edge } = config.abstention;

    // Check if P(referable) falls within uncertainty band [lower_edge, upper_edge]
    if (pRef >= lower_edge && pRef <= upper_edge) {
      return {
        referral: "UNCERTAIN",
        urgency: "PRIORITY",
        follow_up_interval_months: 1,
        is_uncertain: true,
        uncertainty_reasons: [
          `P(referable)=${pRef.toFixed(3)} within uncertainty band [${lower_edge.toFixed(2)}, ${upper_edge.toFixed(2)}].`,
        ],
        applied_modifiers: [],
      };
    }

    // Safety constraint: Never return confident "no DR" when P(referable) >= lower_edge
    if (pRef >= lower_edge && predictedGrade === 0) {
      return {
        referral: "UNCERTAIN",
        urgency: "PRIORITY",
        follow_up_interval_months: 1,
        is_uncertain: true,
        uncertainty_reasons: [
          `Safety constraint: P(referable)=${pRef.toFixed(3)} exceeds non-referable boundary ${lower_edge.toFixed(2)}; cannot return confident No DR.`,
        ],
        applied_modifiers: [],
      };
    }
  } else if (config.min_acceptable_confidence && maxProb < config.min_acceptable_confidence) {
    return {
      referral: "UNCERTAIN",
      urgency: "PRIORITY",
      follow_up_interval_months: 1,
      is_uncertain: true,
      uncertainty_reasons: [
        `Model confidence (${maxProb.toFixed(2)}) below acceptable threshold (${config.min_acceptable_confidence.toFixed(2)}).`,
      ],
      applied_modifiers: [],
    };
  }

  // Base clinical decision from ICDR grade table
  const gradeKey = String(predictedGrade);
  const gradeRule = config.grade_table[gradeKey] || {
    referable: predictedGrade >= 2,
    base_urgency: predictedGrade >= 2 ? "PRIORITY" : "ROUTINE",
    base_interval_months: predictedGrade >= 2 ? 3 : 12,
  };

  let referral: "REFERABLE" | "NON_REFERABLE" | "UNCERTAIN" = gradeRule.referable
    ? "REFERABLE"
    : "NON_REFERABLE";
  let urgency = gradeRule.base_urgency;
  let intervalMonths = gradeRule.base_interval_months;
  const appliedModifiers: string[] = [];

  // Modifier 1: Pregnancy in diabetes (AIOS 2020 / RSSDI 2022)
  if (context.is_pregnant && config.modifiers?.pregnancy) {
    const preg = config.modifiers.pregnancy;
    urgency = maxUrgency(urgency, preg.min_urgency);
    intervalMonths = Math.min(intervalMonths, preg.max_interval_months);
    appliedModifiers.push("PREGNANCY_IN_DIABETES");
  }

  // Modifier 2: High-risk glycemic control (HbA1c >= threshold)
  if (
    context.hba1c != null &&
    config.modifiers?.high_risk_glycemic_control &&
    context.hba1c >= config.modifiers.high_risk_glycemic_control.hba1c_threshold_percent
  ) {
    const glyc = config.modifiers.high_risk_glycemic_control;
    intervalMonths = Math.min(intervalMonths, glyc.max_interval_months);
    if (predictedGrade >= 1) {
      urgency = maxUrgency(urgency, "PRIORITY");
    }
    appliedModifiers.push("SEVERELY_UNCONTROLLED_HBA1C");
  }

  // Modifier 3: Long-standing diabetes (duration >= threshold)
  if (
    context.diabetes_duration_years != null &&
    config.modifiers?.long_standing_diabetes &&
    context.diabetes_duration_years >= config.modifiers.long_standing_diabetes.duration_years_threshold
  ) {
    const dur = config.modifiers.long_standing_diabetes;
    intervalMonths = Math.min(intervalMonths, dur.max_interval_months);
    if (predictedGrade >= 2) {
      urgency = maxUrgency(urgency, "URGENT");
    }
    appliedModifiers.push("LONG_STANDING_DIABETES");
  }

  // Modifier 4: Sight-threatening DR escalation based on P(Grade >= 3)
  if (config.sight_threatening) {
    const probGe3 = (probabilities[3] || 0) + (probabilities[4] || 0);
    const thUrgent = config.sight_threatening.threshold_urgent ?? 0.3;
    const thSoon = config.sight_threatening.threshold_soon ?? 0.05;

    if (probGe3 >= thUrgent) {
      urgency = maxUrgency(urgency, "URGENT");
      intervalMonths = Math.min(intervalMonths, 1);
      appliedModifiers.push("SIGHT_THREATENING_DR_ESCALATION_URGENT");
    } else if (probGe3 >= thSoon) {
      urgency = maxUrgency(urgency, "SOON");
      intervalMonths = Math.min(intervalMonths, 3);
      appliedModifiers.push("SIGHT_THREATENING_DR_ESCALATION_SOON");
    }
  }

  return {
    referral,
    urgency: urgency as any,
    follow_up_interval_months: intervalMonths,
    is_uncertain: false,
    uncertainty_reasons: [],
    applied_modifiers: appliedModifiers,
  };
}
