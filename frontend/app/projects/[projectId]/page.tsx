import { ProjectView } from "@/components/ProjectView";

export default async function ProjectPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  return <ProjectView key={projectId} projectId={projectId} />;
}
