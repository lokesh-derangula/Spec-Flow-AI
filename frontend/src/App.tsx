import { useState } from 'react';
import Dashboard from './components/Dashboard';
import GeneratorStudio from './components/GeneratorStudio';
import CicdHub from './components/CicdHub';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [apiKey, setApiKey] = useState<string>('');
  
  // Shared state to link generator output directly to other workspaces
  const [generatedSpec, setGeneratedSpec] = useState<{
    gherkin: string;
    pageCode: string;
    pageFilename: string;
    specCode: string;
    specFilename: string;
  } | null>(null);

  const handleGenerated = (data: {
    gherkin: string;
    pageCode: string;
    pageFilename: string;
    specCode: string;
    specFilename: string;
  }) => {
    setGeneratedSpec(data);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard onStart={() => setActiveTab('generator')} />;
      case 'generator':
        return (
          <GeneratorStudio 
            apiKey={apiKey} 
            setApiKey={setApiKey} 
            onGenerated={handleGenerated} 
            generatedSpec={generatedSpec}
          />
        );
      case 'cicd':
        return <CicdHub />;
      default:
        return <Dashboard onStart={() => setActiveTab('generator')} />;
    }
  };

  const navItems = [
    { id: 'dashboard', label: 'Home', index: '01' },
    { id: 'generator', label: 'Magic Studio', index: '02' },
    { id: 'cicd', label: 'Integrations', index: '03' },
  ];

  return (
    <div className="saas-app-card">
      {/* Top Header Navbar matching NixtNode layout */}
      <header className="saas-header">
        {/* Left Side: Brand Logo with brace formatting */}
        <div className="saas-logo-container" onClick={() => setActiveTab('dashboard')}>
          <span className="saas-logo-brace">{"}"}</span>
          <span>SpecFlowAI</span>
        </div>

        {/* Center: Navigation Links with superscript indexes */}
        <nav className="saas-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`saas-nav-link ${activeTab === item.id ? 'active' : ''}`}
            >
              {item.label} <sup>{item.index}</sup>
            </button>
          ))}
        </nav>

      </header>

      {/* Main App Body */}
      <main className="saas-body">
        {renderContent()}
      </main>

      {/* Monospace System status line footer */}
      <div className="promo-ribbon">
        <span>STATUS: <strong>ONLINE</strong> &nbsp;|&nbsp; LOCAL NLP ENGINE: <strong>OPTIMAL (T5)</strong> &nbsp;|&nbsp; PARALLEL RUNNER: <strong>CONNECTED</strong></span>
      </div>
    </div>
  );
}
