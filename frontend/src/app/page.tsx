"use client";

import { useState, useEffect } from 'react';
import { Tabs, Card, Statistic, Row, Col, Button, Space, Alert, ConfigProvider, Progress, Spin, message } from 'antd';
import { 
  DashboardOutlined, 
  CodeOutlined, 
  RobotOutlined, 
  NodeIndexOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { auraAPI, type SystemHealthResponse, type TDAResponse, type LNNResponse, type UnifiedAnalysisResponse, type AgentNetwork } from '@/lib/aura-api';

export default function AURADashboard() {
  const [systemHealth, setSystemHealth] = useState<SystemHealthResponse | null>(null);
  const [tdaResults, setTdaResults] = useState<TDAResponse | null>(null);
  const [lnnResults, setLnnResults] = useState<LNNResponse | null>(null);
  const [unifiedResults, setUnifiedResults] = useState<UnifiedAnalysisResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentNetwork, setAgentNetwork] = useState<AgentNetwork[]>([
    { id: '1', name: 'ResourceAllocator', status: 'active', connections: 1247, performance: 0.95, last_activity: new Date().toISOString() },
    { id: '2', name: 'RiskAssessor', status: 'active', connections: 892, performance: 0.88, last_activity: new Date().toISOString() },
    { id: '3', name: 'PolicyEnforcer', status: 'active', connections: 1563, performance: 0.92, last_activity: new Date().toISOString() }
  ]);

  // Initialize health monitoring
  useEffect(() => {
    // Initial health check
    checkSystemHealth();
    
    // Start health monitoring
    auraAPI.startHealthMonitoring((health) => {
      setSystemHealth(health);
    }, 3000);

    return () => {
      auraAPI.stopHealthMonitoring();
    };
  }, []);

  const checkSystemHealth = async () => {
    const health = await auraAPI.getSystemHealth();
    setSystemHealth(health);
  };

  const runTDAnalysis = async () => {
    setIsProcessing(true);
    message.info('Running Topological Data Analysis...');
    
    try {
      const sampleData = auraAPI.generateSampleData();
      const result = await auraAPI.analyzeTopology(sampleData, 2);
      setTdaResults(result);
      
      if (result.success) {
        message.success(`TDA Analysis completed in ${result.processing_time_ms}ms`);
      } else {
        message.error(`TDA Analysis failed: ${result.error}`);
      }
    } catch (error) {
      message.error('TDA Analysis failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const runLNNProcessing = async () => {
    setIsProcessing(true);
    message.info('Running Liquid Neural Network processing...');
    
    try {
      const sampleMatrix = auraAPI.generateSampleMatrix();
      const result = await auraAPI.processLNN(sampleMatrix, 10, 64);
      setLnnResults(result);
      
      if (result.success) {
        message.success(`LNN Processing completed in ${result.processing_time_ms}ms`);
      } else {
        message.error(`LNN Processing failed: ${result.error}`);
      }
    } catch (error) {
      message.error('LNN Processing failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const runUnifiedAnalysis = async () => {
    setIsProcessing(true);
    message.info('Running Unified AURA Analysis...');
    
    try {
      const sampleData = {
        topology: auraAPI.generateSampleData(),
        neural: auraAPI.generateSampleMatrix(),
        metadata: {
          timestamp: new Date().toISOString(),
          source: 'dashboard'
        }
      };
      
      const result = await auraAPI.performUnifiedAnalysis(sampleData, ['tda', 'lnn']);
      setUnifiedResults(result);
      
      if (result.success) {
        message.success(`Unified Analysis completed in ${result.processing_time_ms}ms`);
      } else {
        message.error(`Unified Analysis failed: ${result.error}`);
      }
    } catch (error) {
      message.error('Unified Analysis failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const items = [
    {
      key: 'overview',
      label: (
        <span>
          <DashboardOutlined />
          System Overview
        </span>
      ),
      children: <SystemOverviewPanel systemHealth={systemHealth} onRefresh={checkSystemHealth} />
    },
    {
      key: 'tda',
      label: (
        <span>
          <CodeOutlined />
          TDA Analysis
        </span>
      ),
      children: <TDAnalysisPanel results={tdaResults} onRun={runTDAnalysis} isProcessing={isProcessing} />
    },
    {
      key: 'lnn',
      label: (
        <span>
          <RobotOutlined />
          LNN Processing
        </span>
      ),
      children: <LNNProcessingPanel results={lnnResults} onRun={runLNNProcessing} isProcessing={isProcessing} />
    },
    {
      key: 'unified',
      label: (
        <span>
          <NodeIndexOutlined />
          Unified Analysis
        </span>
      ),
      children: <UnifiedAnalysisPanel results={unifiedResults} onRun={runUnifiedAnalysis} isProcessing={isProcessing} />
    },
    {
      key: 'agents',
      label: (
        <span>
          <ThunderboltOutlined />
          Agent Network
        </span>
      ),
      children: <AgentNetworkPanel agents={agentNetwork} />
    },
    {
      key: 'monitoring',
      label: (
        <span>
          <LineChartOutlined />
          Real-time Monitoring
        </span>
      ),
      children: <RealTimeMonitoringPanel systemHealth={systemHealth} />
    }
  ];

  return (
    <ConfigProvider
      theme={{
        token: {
          // Dark theme for AURA
          colorPrimary: '#00ffff',
          colorSuccess: '#00ff00',
          colorWarning: '#ffff00',
          colorError: '#ff0000',
          colorInfo: '#00ffff',
          
          // Dark background
          colorBgContainer: '#000000',
          colorBgElevated: '#0a0a0a',
          colorBgLayout: '#000000',
          colorBgSpotlight: '#141414',
          
          // Text colors
          colorText: '#ffffff',
          colorTextSecondary: '#cccccc',
          colorTextTertiary: '#999999',
          
          // Border colors
          colorBorder: '#333333',
          colorBorderSecondary: '#222222',
        },
      }}
    >
      <div className="min-h-screen aura-bg text-white">
        {/* Header */}
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-3xl font-bold text-cyan-400 mb-2 aura-text-glow">
            AURA System Command Center
          </h1>
          <p className="text-gray-400">
            Real-time Unified Geometric Platform & AI Intelligence System
          </p>
          {systemHealth && (
            <div className="mt-4 flex items-center gap-4">
              <span className={`px-2 py-1 rounded text-sm ${
                systemHealth.status === 'healthy' ? 'bg-green-500 text-black' : 'bg-red-500 text-white'
              }`}>
                {systemHealth.status.toUpperCase()}
              </span>
              <span className="text-gray-400">
                Uptime: {Math.floor(systemHealth.uptime_seconds / 3600)}h {Math.floor((systemHealth.uptime_seconds % 3600) / 60)}m
              </span>
              <span className="text-gray-400">
                Requests: {systemHealth.total_requests} ({(systemHealth.successful_requests / systemHealth.total_requests * 100).toFixed(1)}% success)
              </span>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="p-6">
          <Tabs
            items={items}
            type="card"
            className="aura-tabs"
            tabBarStyle={{ 
              borderBottom: '1px solid #333',
              marginBottom: 24 
            }}
          />
        </div>
      </div>
    </ConfigProvider>
  );
}

// System Overview Panel
function SystemOverviewPanel({ systemHealth, onRefresh }: { systemHealth: SystemHealthResponse | null, onRefresh: () => void }) {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <Alert
          message="AURA System Status"
          description="Real-time system health and performance metrics"
          type="info"
          showIcon
          className="glass-panel"
        />
        <Button icon={<ReloadOutlined />} onClick={onRefresh}>
          Refresh
        </Button>
      </div>

      {systemHealth ? (
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} lg={6}>
            <Card className="glass-panel">
              <Statistic
                title="System Status"
                value={systemHealth.status}
                valueStyle={{ 
                  color: systemHealth.status === 'healthy' ? '#00ff00' : '#ff0000' 
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card className="glass-panel">
              <Statistic
                title="Total Requests"
                value={systemHealth.total_requests}
                valueStyle={{ color: '#00ffff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card className="glass-panel">
              <Statistic
                title="Success Rate"
                value={(systemHealth.successful_requests / systemHealth.total_requests * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: '#00ff00' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card className="glass-panel">
              <Statistic
                title="Avg Response Time"
                value={systemHealth.average_processing_time_ms.toFixed(1)}
                suffix="ms"
                valueStyle={{ color: '#ffff00' }}
              />
            </Card>
          </Col>
        </Row>
      ) : (
        <Card className="glass-panel">
          <div className="text-center py-8">
            <Spin size="large" />
            <p className="mt-4 text-gray-400">Connecting to AURA system...</p>
          </div>
        </Card>
      )}
    </div>
  );
}

// TDA Analysis Panel
function TDAnalysisPanel({ results, onRun, isProcessing }: { results: TDAResponse | null, onRun: () => void, isProcessing: boolean }) {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <Alert
          message="Topological Data Analysis"
          description="Real-time topological analysis using persistent homology"
          type="info"
          showIcon
          className="glass-panel"
        />
        <Button 
          type="primary" 
          icon={<PlayCircleOutlined />} 
          onClick={onRun}
          loading={isProcessing}
        >
          Run TDA Analysis
        </Button>
      </div>

      {results && (
        <Card title="Analysis Results" className="glass-panel">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>Status:</span>
              <span className={results.success ? 'text-green-400' : 'text-red-400'}>
                {results.success ? 'Success' : 'Failed'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Processing Time:</span>
              <span className="text-cyan-400">{results.processing_time_ms}ms</span>
            </div>
            {results.betti_numbers && (
              <div>
                <h4 className="text-lg font-semibold mb-2">Betti Numbers:</h4>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(results.betti_numbers).map(([dim, value]) => (
                    <div key={dim} className="text-center">
                      <div className="text-2xl font-bold text-cyan-400">β{dim}</div>
                      <div className="text-sm text-gray-400">{value}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {results.error && (
              <Alert message="Error" description={results.error} type="error" showIcon />
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

// LNN Processing Panel
function LNNProcessingPanel({ results, onRun, isProcessing }: { results: LNNResponse | null, onRun: () => void, isProcessing: boolean }) {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <Alert
          message="Liquid Neural Network Processing"
          description="Real-time neural network processing with MIT LNN"
          type="info"
          showIcon
          className="glass-panel"
        />
        <Button 
          type="primary" 
          icon={<PlayCircleOutlined />} 
          onClick={onRun}
          loading={isProcessing}
        >
          Run LNN Processing
        </Button>
      </div>

      {results && (
        <Card title="Processing Results" className="glass-panel">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>Status:</span>
              <span className={results.success ? 'text-green-400' : 'text-red-400'}>
                {results.success ? 'Success' : 'Failed'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Processing Time:</span>
              <span className="text-cyan-400">{results.processing_time_ms}ms</span>
            </div>
            {results.output && (
              <div>
                <h4 className="text-lg font-semibold mb-2">Output Shape:</h4>
                <div className="text-cyan-400">
                  {results.output.length} × {results.output[0]?.length || 0}
                </div>
              </div>
            )}
            {results.error && (
              <Alert message="Error" description={results.error} type="error" showIcon />
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

// Unified Analysis Panel
function UnifiedAnalysisPanel({ results, onRun, isProcessing }: { results: UnifiedAnalysisResponse | null, onRun: () => void, isProcessing: boolean }) {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <Alert
          message="Unified AURA Analysis"
          description="Combined TDA + LNN analysis using unified system"
          type="info"
          showIcon
          className="glass-panel"
        />
        <Button 
          type="primary" 
          icon={<PlayCircleOutlined />} 
          onClick={onRun}
          loading={isProcessing}
        >
          Run Unified Analysis
        </Button>
      </div>

      {results && (
        <Card title="Unified Results" className="glass-panel">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>Status:</span>
              <span className={results.success ? 'text-green-400' : 'text-red-400'}>
                {results.success ? 'Success' : 'Failed'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Processing Time:</span>
              <span className="text-cyan-400">{results.processing_time_ms}ms</span>
            </div>
            {results.results && Object.keys(results.results).length > 0 && (
              <div>
                <h4 className="text-lg font-semibold mb-2">Analysis Results:</h4>
                <pre className="bg-gray-900 p-4 rounded text-sm overflow-auto">
                  {JSON.stringify(results.results, null, 2)}
                </pre>
              </div>
            )}
            {results.error && (
              <Alert message="Error" description={results.error} type="error" showIcon />
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

// Agent Network Panel
function AgentNetworkPanel({ agents }: { agents: AgentNetwork[] }) {
  return (
    <div className="space-y-6">
      <Alert
        message="Agent Network Status"
        description="Real-time agent network monitoring and performance"
        type="info"
        showIcon
        className="glass-panel"
      />
      
      <Row gutter={[16, 16]}>
        {agents.map(agent => (
          <Col xs={24} sm={12} lg={8} key={agent.id}>
            <Card 
              title={agent.name} 
              className="glass-panel neural-glow"
              extra={
                <div className={`w-2 h-2 rounded-full ${
                  agent.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                }`} />
              }
            >
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Connections:</span>
                  <span className="text-cyan-400">{agent.connections}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Performance:</span>
                  <span className="text-green-400">{(agent.performance * 100).toFixed(1)}%</span>
                </div>
                <Progress 
                  percent={agent.performance * 100} 
                  strokeColor="#00ffff"
                  showInfo={false}
                />
                <div className="text-xs text-gray-400">
                  Last Activity: {new Date(agent.last_activity).toLocaleTimeString()}
                </div>
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
}

// Real-time Monitoring Panel
function RealTimeMonitoringPanel({ systemHealth }: { systemHealth: SystemHealthResponse | null }) {
  return (
    <div className="space-y-6">
      <Alert
        message="Real-time System Monitoring"
        description="Live system metrics and performance tracking"
        type="info"
        showIcon
        className="glass-panel"
      />
      
      {systemHealth ? (
        <div className="grid grid-cols-2 gap-6">
          <Card title="Request Success Rate" className="glass-panel">
            <div className="text-center">
              <div className="text-4xl font-bold text-green-400">
                {((systemHealth.successful_requests / systemHealth.total_requests) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-400 mt-2">
                {systemHealth.successful_requests} / {systemHealth.total_requests} requests
              </div>
            </div>
          </Card>
          
          <Card title="Average Response Time" className="glass-panel">
            <div className="text-center">
              <div className="text-4xl font-bold text-cyan-400">
                {systemHealth.average_processing_time_ms.toFixed(1)}ms
              </div>
              <div className="text-sm text-gray-400 mt-2">
                Processing time
              </div>
            </div>
          </Card>
          
          <Card title="System Uptime" className="glass-panel">
            <div className="text-center">
              <div className="text-4xl font-bold text-yellow-400">
                {Math.floor(systemHealth.uptime_seconds / 3600)}h {Math.floor((systemHealth.uptime_seconds % 3600) / 60)}m
              </div>
              <div className="text-sm text-gray-400 mt-2">
                System running time
              </div>
            </div>
          </Card>
          
          <Card title="TDA Engine Status" className="glass-panel">
            <div className="text-center">
              <div className={`text-4xl font-bold ${
                systemHealth.tda_engine_status === 'online' ? 'text-green-400' : 'text-red-400'
              }`}>
                {systemHealth.tda_engine_status.toUpperCase()}
              </div>
              <div className="text-sm text-gray-400 mt-2">
                Topology engine
              </div>
            </div>
          </Card>
        </div>
      ) : (
        <Card className="glass-panel">
          <div className="text-center py-8">
            <Spin size="large" />
            <p className="mt-4 text-gray-400">Loading system metrics...</p>
          </div>
        </Card>
      )}
    </div>
  );
}
