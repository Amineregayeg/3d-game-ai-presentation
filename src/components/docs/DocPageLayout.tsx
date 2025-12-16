"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

interface Phase {
  name: string;
  duration: string;
  tasks: string[];
  milestone?: string;
}

interface Section {
  title: string;
  content: string;
  code?: string;
  language?: string;
}

interface DocPageLayoutProps {
  title: string;
  subtitle: string;
  description: string;
  gradient: string;
  accentColor: string;
  icon: React.ReactNode;
  presentationLink: string;
  technologies: string[];
  phases: Phase[];
  sections: Section[];
  apiReference?: {
    endpoint: string;
    method: string;
    description: string;
    parameters?: { name: string; type: string; description: string }[];
  }[];
}

export function DocPageLayout({
  title,
  subtitle,
  description,
  gradient,
  accentColor,
  icon,
  presentationLink,
  technologies,
  phases,
  sections,
  apiReference,
}: DocPageLayoutProps) {
  const totalTasks = phases.reduce((acc, phase) => acc + phase.tasks.length, 0);
  const completedTasks = 0; // In production, this would be from state/DB

  return (
    <div className="dark min-h-screen bg-slate-950">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-1/4 -left-32 w-96 h-96 ${accentColor}/10 rounded-full blur-3xl`} />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-xl sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/docs"
                className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                All Docs
              </Link>
              <div className="w-px h-6 bg-slate-700" />
              <div className="flex items-center gap-2">
                <div className={`p-1.5 rounded-lg bg-gradient-to-br ${gradient}`}>
                  <div className="text-white w-4 h-4 [&>svg]:w-4 [&>svg]:h-4">{icon}</div>
                </div>
                <span className="text-white font-medium">{title}</span>
              </div>
            </div>
            <Button asChild className={`bg-gradient-to-r ${gradient} text-white hover:opacity-90`}>
              <Link href={presentationLink}>
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                View Presentation
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Hero Section */}
        <div className="mb-8">
          <div className="flex items-start gap-4 mb-4">
            <div className={`p-4 rounded-2xl bg-gradient-to-br ${gradient}`}>
              <div className="text-white">{icon}</div>
            </div>
            <div>
              <h1 className={`text-4xl font-bold bg-gradient-to-r ${gradient} bg-clip-text text-transparent mb-2`}>
                {title}
              </h1>
              <p className="text-xl text-slate-400">{subtitle}</p>
            </div>
          </div>
          <p className="text-slate-400 max-w-3xl">{description}</p>
        </div>

        {/* Technologies */}
        <div className="flex flex-wrap gap-2 mb-8">
          {technologies.map((tech) => (
            <Badge key={tech} variant="secondary" className="bg-slate-800/50 text-slate-300 border border-slate-700">
              {tech}
            </Badge>
          ))}
        </div>

        {/* Progress Overview */}
        <Card className="bg-slate-900/50 border-slate-800 mb-8">
          <CardHeader className="pb-2">
            <CardTitle className="text-white text-lg">Implementation Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between text-sm text-slate-400 mb-2">
              <span>{completedTasks} of {totalTasks} tasks completed</span>
              <span>{Math.round((completedTasks / totalTasks) * 100)}%</span>
            </div>
            <Progress value={(completedTasks / totalTasks) * 100} className="h-2" />
          </CardContent>
        </Card>

        {/* Main Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="bg-slate-900/50 border border-slate-800">
            <TabsTrigger value="overview" className="data-[state=active]:bg-slate-800">Overview</TabsTrigger>
            <TabsTrigger value="architecture" className="data-[state=active]:bg-slate-800">Architecture</TabsTrigger>
            <TabsTrigger value="implementation" className="data-[state=active]:bg-slate-800">Implementation</TabsTrigger>
            {apiReference && (
              <TabsTrigger value="api" className="data-[state=active]:bg-slate-800">API Reference</TabsTrigger>
            )}
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-6">
                {sections.slice(0, 3).map((section, index) => (
                  <Card key={index} className="bg-slate-900/50 border-slate-800">
                    <CardHeader>
                      <CardTitle className="text-white">{section.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-slate-400 whitespace-pre-line">{section.content}</p>
                      {section.code && (
                        <pre className="mt-4 p-4 bg-slate-950 rounded-lg overflow-x-auto">
                          <code className="text-sm text-slate-300">{section.code}</code>
                        </pre>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
              <div className="space-y-6">
                <Card className="bg-slate-900/50 border-slate-800">
                  <CardHeader>
                    <CardTitle className="text-white">Quick Stats</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <p className="text-xs text-slate-500 mb-1">Total Phases</p>
                      <p className="text-2xl font-bold text-white">{phases.length}</p>
                    </div>
                    <Separator className="bg-slate-800" />
                    <div>
                      <p className="text-xs text-slate-500 mb-1">Total Tasks</p>
                      <p className="text-2xl font-bold text-white">{totalTasks}</p>
                    </div>
                    <Separator className="bg-slate-800" />
                    <div>
                      <p className="text-xs text-slate-500 mb-1">Technologies</p>
                      <p className="text-2xl font-bold text-white">{technologies.length}</p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Architecture Tab */}
          <TabsContent value="architecture" className="space-y-6">
            <ScrollArea className="h-[600px]">
              <div className="space-y-6 pr-4">
                {sections.map((section, index) => (
                  <Card key={index} className="bg-slate-900/50 border-slate-800">
                    <CardHeader>
                      <CardTitle className="text-white">{section.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-slate-400 whitespace-pre-line">{section.content}</p>
                      {section.code && (
                        <pre className="mt-4 p-4 bg-slate-950 rounded-lg overflow-x-auto">
                          <code className="text-sm text-slate-300">{section.code}</code>
                        </pre>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Implementation Tab */}
          <TabsContent value="implementation" className="space-y-6">
            <Accordion type="single" collapsible className="space-y-4">
              {phases.map((phase, index) => (
                <AccordionItem
                  key={index}
                  value={`phase-${index}`}
                  className="border border-slate-800 rounded-lg bg-slate-900/50 px-4"
                >
                  <AccordionTrigger className="hover:no-underline py-4">
                    <div className="flex items-center gap-4">
                      <div className={`w-8 h-8 rounded-full bg-gradient-to-br ${gradient} flex items-center justify-center text-white font-bold text-sm`}>
                        {index + 1}
                      </div>
                      <div className="text-left">
                        <p className="text-white font-medium">{phase.name}</p>
                        <p className="text-xs text-slate-500">{phase.duration}</p>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4">
                    <div className="ml-12 space-y-3">
                      {phase.tasks.map((task, taskIndex) => (
                        <div key={taskIndex} className="flex items-start gap-3 text-sm">
                          <div className="w-5 h-5 rounded border border-slate-700 flex-shrink-0 mt-0.5" />
                          <span className="text-slate-400">{task}</span>
                        </div>
                      ))}
                      {phase.milestone && (
                        <div className="mt-4 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                          <p className="text-xs text-slate-500 mb-1">Milestone</p>
                          <p className="text-sm text-emerald-400">{phase.milestone}</p>
                        </div>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </TabsContent>

          {/* API Reference Tab */}
          {apiReference && (
            <TabsContent value="api" className="space-y-6">
              <ScrollArea className="h-[600px]">
                <div className="space-y-4 pr-4">
                  {apiReference.map((api, index) => (
                    <Card key={index} className="bg-slate-900/50 border-slate-800">
                      <CardHeader className="pb-2">
                        <div className="flex items-center gap-2">
                          <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                            {api.method}
                          </Badge>
                          <code className="text-white font-mono">{api.endpoint}</code>
                        </div>
                        <CardDescription className="text-slate-400 mt-2">
                          {api.description}
                        </CardDescription>
                      </CardHeader>
                      {api.parameters && api.parameters.length > 0 && (
                        <CardContent>
                          <p className="text-xs text-slate-500 mb-2">Parameters</p>
                          <div className="space-y-2">
                            {api.parameters.map((param, paramIndex) => (
                              <div key={paramIndex} className="flex items-start gap-3 text-sm">
                                <code className="text-cyan-400 font-mono">{param.name}</code>
                                <Badge variant="secondary" className="bg-slate-800 text-slate-400 text-xs">
                                  {param.type}
                                </Badge>
                                <span className="text-slate-500">{param.description}</span>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      )}
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>
          )}
        </Tabs>
      </main>
    </div>
  );
}
