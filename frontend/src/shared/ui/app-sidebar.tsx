import type { FC } from 'react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarHeader,
} from '../ui/sidebar';
import AILogo from './ai-logo';

export const AppSidebar: FC = () => (
  <Sidebar>
    <SidebarHeader>
      <AILogo className='text-[30px]' size={30} />
    </SidebarHeader>
    <SidebarContent>
      <SidebarGroup />

      <SidebarGroup />
    </SidebarContent>
    <SidebarFooter />
  </Sidebar>
);
