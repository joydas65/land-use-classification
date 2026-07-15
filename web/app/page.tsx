import type { Metadata } from "next";
import { TerraClassApp } from "./TerraClassApp";

export const metadata: Metadata = {
  title: "TerraClass | Satellite Land-Use Classification",
  description:
    "Explore a leakage-aware ResNet18 classifier for five UC Merced land-use classes.",
};

export default function Home() {
  return <TerraClassApp />;
}
