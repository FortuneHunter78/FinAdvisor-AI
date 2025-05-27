import type { FC, PropsWithChildren } from 'react';
import { AppSidebar } from '../ui/app-sidebar';
import { SidebarProvider, SidebarTrigger } from '../ui/sidebar';
import HoverInfoCard from '../ui/hover-info-card';
import { CircleHelp, CircleUserRound } from 'lucide-react';
import LinkButton from '../ui/link-button';

const MainLayout: FC<PropsWithChildren> = ({ children }) => (
  <SidebarProvider>
    <AppSidebar />

    <div className='w-full'>
      <header className='w-full h-[50px] border-b border-b-sidebar-border flex justify-between md:pr-5 p-2'>
        <SidebarTrigger />

        <div className='flex gap-2'>
          <h3 className='text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500'>
            Chart.AI
          </h3>

          <HoverInfoCard
            info={
              <p className='text-center'>
                <span className='font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500'>
                  FinadvisorAI
                </span>{' '}
                - платформа, где машинное обучение помогает вам принимать
                обоснованные инвестиционные решения. Мы предлагаем инструменты
                для анализа финансовых данных и получения полезных рекомендаций.
              </p>
            }
          />
        </div>

        <div className='flex gap-1 items-center '>
          <HoverInfoCard
            icon={<CircleHelp size={20} />}
            info={
              <p>
                Авторизация дает возможность сохранять историю прогнозов и
                многое другое
              </p>
            }
          />

          <LinkButton to='/login' variant='outline'>
            <p className='md:block hidden'>Войти</p>
            <CircleUserRound />
          </LinkButton>
        </div>
      </header>

      <main className='flex justify-center items-center h-[calc(100%-50px)] p-5'>
        {children}
      </main>
    </div>
  </SidebarProvider>
);

export default MainLayout;
